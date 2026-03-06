#!/usr/bin/env python3
"""
auto_video_edit.py (streaming encode + swing + block detection)

Purpose
-------
Automatically extract cricket "shot moments" from a static-camera practice video where
the batsman is centered (indoor nets). Outputs a highlight video containing only shots.

Detection
---------
Two complementary detectors (both motion-based in an ROI):
1) Burst/Peak detector (good for full swings): looks for sharp motion spikes (robust z-score).
2) Energy detector (good for dead-bat blocks + soft pushes): looks for sustained elevated motion
   over a short window (robust z-score of window-mean + a minimum floor).

Speed
-----
- Starts encoding while scanning (producer/consumer pipeline).
- NVIDIA NVENC encoding via FFmpeg (--gpu) + optional CUDA decode (--hwaccel).
- Parallel segment encoding (--jobs).
- Fast concat without re-encode (--concat_copy).

Requirements
------------
- ffmpeg + ffprobe in PATH (gyan.dev build is great)
- pip install opencv-python numpy tqdm

Examples
--------
Test first 2 minutes:
  python auto_video_edit.py -i "Timeline 1.mov" -o highlights_test.mp4 --scan 120 --gpu --hwaccel --jobs 3 --concat_copy

Full video:
  python auto_video_edit.py -i "Timeline 1.mov" -o highlights_full.mp4 --gpu --hwaccel --jobs 3 --concat_copy

If it misses dead-bat blocks:
  --energy_z 1.1 --energy_min 0.002

If it includes too much junk:
  --energy_min 0.003 --peak_z 2.7
"""
import argparse
import subprocess
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n\n"
            f"STDOUT:\n{p.stdout}\n\n"
            f"STDERR:\n{p.stderr}\n"
        )


def build_segment_cmd(
    input_path: str,
    start_s: float,
    dur_s: float,
    out_path: str,
    *,
    use_gpu: bool,
    use_hwaccel: bool,
    gpu_codec: str,
    nvenc_cq: int,
    cpu_crf: int,
    cpu_preset: str,
    audio_bitrate: str,
) -> list[str]:
    nvenc_map = {"h264": "h264_nvenc", "hevc": "hevc_nvenc", "av1": "av1_nvenc"}
    if gpu_codec not in nvenc_map:
        raise ValueError("gpu_codec must be one of: h264, hevc, av1")
    nvenc_encoder = nvenc_map[gpu_codec]

    cmd = ["ffmpeg", "-y"]
    if use_hwaccel:
        cmd += ["-hwaccel", "cuda"]

    # Accurate seek: -i then -ss (slower but stable timestamps)
    cmd += [
        "-i", input_path,
        "-ss", f"{start_s:.3f}",
        "-t", f"{dur_s:.3f}",
        "-fflags", "+genpts",
        "-avoid_negative_ts", "make_zero",
    ]

    if use_gpu:
        cmd += [
            "-c:v", nvenc_encoder,
            "-preset", "p5",
            "-cq", str(nvenc_cq),
            "-rc", "vbr",
            "-b:v", "0",
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", cpu_preset,
            "-crf", str(cpu_crf),
        ]

    cmd += [
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        out_path,
    ]
    return cmd


def stream_detect_and_encode(
    video_path: str,
    output_path: str,
    *,
    scan_seconds: float | None,
    sample_fps: float,
    roi: tuple[float, float, float, float],
    smooth_window: int,
    peak_z: float,
    min_gap_sec: float,
    pre: float,
    post: float,
    merge_gap: float,
    # Energy detector params (blocks/pushes)
    energy_window: float,
    energy_z: float,
    energy_min: float,
    # Encoding params
    jobs: int,
    use_gpu: bool,
    use_hwaccel: bool,
    gpu_codec: str,
    nvenc_cq: int,
    cpu_crf: int,
    cpu_preset: str,
    audio_bitrate: str,
    concat_copy: bool,
) -> None:
    video_path_abs = str(Path(video_path).resolve())
    output_path_abs = str(Path(output_path).resolve())

    cap = cv2.VideoCapture(video_path_abs)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path_abs}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / max(sample_fps, 1.0))))
    effective_scan_fps = native_fps / step

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if scan_seconds is None:
        scan_frames_limit = total_frames
    else:
        scan_frames_limit = min(total_frames, int(scan_seconds * native_fps))

    # ROI bounds sanity
    x1, y1, x2, y2 = roi
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("ROI must be normalized floats 0..1 with x1<x2 and y1<y2")

    # Background subtractor (static camera)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

    # Rolling motion history for robust stats
    hist_len = max(30, int(effective_scan_fps * 6))  # ~6 seconds
    motion_hist = deque(maxlen=hist_len)

    # Rolling energy history (for robust energy z-score)
    energy_hist = deque(maxlen=max(30, int(effective_scan_fps * 10)))  # ~10 seconds
    energy_frames = max(3, int(round(energy_window * effective_scan_fps)))

    # Burst detection state (swing spikes)
    in_burst = False
    burst_max_z = -1e9
    burst_max_t = 0.0

    last_event_t = -1e9

    # Pending merged segment not yet submitted
    pending: list[float] | None = None
    pending_idx = 0

    workers = max(1, int(jobs))
    futures = []
    part_paths: dict[int, Path] = {}

    with tempfile.TemporaryDirectory(prefix="shot_keep_stream_") as tmp:
        tmpdir = Path(tmp)
        parts_dir = tmpdir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)

        def submit_range(range_start: float, range_end: float, idx: int):
            s = max(0.0, float(range_start))
            e = max(s + 0.01, float(range_end))
            dur_s = e - s

            part = parts_dir / f"part_{idx:05d}.mp4"
            cmd = build_segment_cmd(
                video_path_abs, s, dur_s, str(part),
                use_gpu=use_gpu,
                use_hwaccel=use_hwaccel,
                gpu_codec=gpu_codec,
                nvenc_cq=nvenc_cq,
                cpu_crf=cpu_crf,
                cpu_preset=cpu_preset,
                audio_bitrate=audio_bitrate,
            )
            run(cmd)
            return idx, part

        def enqueue_clip(rs: float, re: float):
            nonlocal pending, pending_idx, futures
            if pending is None:
                pending = [rs, re]
                return
            if rs <= pending[1] + merge_gap:
                pending[1] = max(pending[1], re)
            else:
                idx = pending_idx
                s, e = pending
                futures.append(ex.submit(submit_range, s, e, idx))
                pending_idx += 1
                pending = [rs, re]

        def flush_pending():
            nonlocal pending, pending_idx, futures
            if pending is not None:
                idx = pending_idx
                s, e = pending
                futures.append(ex.submit(submit_range, s, e, idx))
                pending_idx += 1
                pending = None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            frame_i = 0
            pbar = tqdm(total=max(1, scan_frames_limit // step), desc="Scanning + encoding", unit="frame")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_i >= scan_frames_limit:
                    break

                if frame_i % step != 0:
                    frame_i += 1
                    continue

                t = frame_i / native_fps

                # If we're safely beyond pending end, flush it (prevents over-merging across distant actions)
                if pending is not None and t > (pending[1] + merge_gap):
                    flush_pending()

                h, w = frame.shape[:2]
                X1, Y1, X2, Y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                crop = frame[Y1:Y2, X1:X2]

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                fg = fgbg.apply(gray)
                fg = cv2.medianBlur(fg, 5)
                _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

                score = float(np.count_nonzero(fg)) / float(fg.size)
                motion_hist.append(score)

                # ---- Compute robust z-score for current motion (swing detector) ----
                z = None
                if len(motion_hist) >= max(10, smooth_window):
                    arr = np.array(motion_hist, dtype=np.float32)
                    if smooth_window > 1 and len(arr) >= smooth_window:
                        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
                        arr_s = np.convolve(arr, kernel, mode="same")
                        cur = float(arr_s[-1])
                        med = float(np.median(arr_s))
                        mad = float(np.median(np.abs(arr_s - med)) + 1e-9)
                    else:
                        cur = float(arr[-1])
                        med = float(np.median(arr))
                        mad = float(np.median(np.abs(arr - med)) + 1e-9)
                    z = (cur - med) / (1.4826 * mad)

                # ---- Swing burst detector (sharp spikes) ----
                if z is not None:
                    enter_thr = peak_z
                    exit_thr = peak_z * 0.6

                    if not in_burst:
                        if z > enter_thr and (t - last_event_t) >= min_gap_sec:
                            in_burst = True
                            burst_max_z = z
                            burst_max_t = t
                    else:
                        if z > burst_max_z:
                            burst_max_z = z
                            burst_max_t = t
                        if z < exit_thr:
                            # burst ended -> event at burst max
                            event_t = burst_max_t
                            last_event_t = event_t
                            in_burst = False

                            rs = event_t - pre
                            re = event_t + post
                            enqueue_clip(rs, re)

                # ---- Energy detector (blocks + soft pushes) ----
                if len(motion_hist) >= energy_frames:
                    recent = list(motion_hist)[-energy_frames:]
                    energy = float(np.mean(recent))
                    energy_hist.append(energy)

                    if len(energy_hist) >= 20:
                        eh = np.array(energy_hist, dtype=np.float32)
                        med_e = float(np.median(eh))
                        mad_e = float(np.median(np.abs(eh - med_e)) + 1e-9)
                        z_e = (energy - med_e) / (1.4826 * mad_e)

                        if (z_e > energy_z) and (energy > energy_min) and ((t - last_event_t) >= min_gap_sec):
                            event_t = t
                            last_event_t = event_t

                            rs = event_t - pre
                            re = event_t + post
                            enqueue_clip(rs, re)

                frame_i += 1
                pbar.update(1)

            pbar.close()
            cap.release()

            # Flush pending at end
            flush_pending()

            # Collect encode results
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Finishing encodes", unit="clip"):
                idx, part = fut.result()
                part_paths[idx] = part

        if not part_paths:
            raise RuntimeError(
                "No clips were produced.\n"
                "Try: --peak_z 1.7  --energy_z 1.1  --energy_min 0.002  --sample_fps 12  --roi 0.45,0.05,0.95,0.95"
            )

        # Build concat list in correct order
        list_txt = tmpdir / "list.txt"
        with list_txt.open("w", encoding="utf-8") as f:
            for idx in sorted(part_paths.keys()):
                p = part_paths[idx]
                safe_path = str(p).replace("'", "\\'")
                f.write(f"file '{safe_path}'\n")

        # Concat
        if concat_copy:
            cmd_concat = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_txt), "-c", "copy", output_path_abs]
        else:
            nvenc_map = {"h264": "h264_nvenc", "hevc": "hevc_nvenc", "av1": "av1_nvenc"}
            if use_gpu:
                cmd_concat = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(list_txt),
                    "-c:v", nvenc_map[gpu_codec],
                    "-preset", "p5",
                    "-cq", str(nvenc_cq),
                    "-rc", "vbr",
                    "-b:v", "0",
                    "-c:a", "aac",
                    "-b:a", audio_bitrate,
                    "-movflags", "+faststart",
                    output_path_abs
                ]
            else:
                cmd_concat = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(list_txt),
                    "-c:v", "libx264",
                    "-preset", cpu_preset,
                    "-crf", str(cpu_crf),
                    "-c:a", "aac",
                    "-b:a", audio_bitrate,
                    "-movflags", "+faststart",
                    output_path_abs
                ]
        run(cmd_concat)


def main():
    ap = argparse.ArgumentParser(description="Extract cricket shots (swings + blocks) and export highlights (streaming encode).")
    ap.add_argument("-i", "--input", required=True, help='Input video, e.g. "Timeline 1.mov"')
    ap.add_argument("-o", "--output", required=True, help="Output video, e.g. highlights.mp4")

    # Scan / detection
    ap.add_argument("--scan", type=float, default=None, help="Scan only first N seconds (e.g. 120). Default = full video.")
    ap.add_argument("--sample_fps", type=float, default=12.0, help="Analysis FPS. Default 12 (raise for blocks).")
    ap.add_argument("--roi", type=str, default="0.55,0.10,0.92,0.85", help="ROI normalized x1,y1,x2,y2.")
    ap.add_argument("--peak_z", type=float, default=2.5, help="Swing spike sensitivity (lower=more). Default 2.5")
    ap.add_argument("--min_gap", type=float, default=1.8, help="Min seconds between events. Default 1.8")

    ap.add_argument("--pre", type=float, default=0.8, help="Seconds to keep BEFORE event. Default 0.8")
    ap.add_argument("--post", type=float, default=1.4, help="Seconds to keep AFTER event. Default 1.4")
    ap.add_argument("--merge_gap", type=float, default=0.20, help="Merge segments if gap <= this. Default 0.20")

    # Blocks / soft pushes (energy detector)
    ap.add_argument("--energy_window", type=float, default=0.8, help="Seconds window for energy detection. Default 0.8")
    ap.add_argument("--energy_z", type=float, default=1.3, help="Energy z-threshold (lower=more blocks). Default 1.3")
    ap.add_argument("--energy_min", type=float, default=0.0025, help="Min avg motion in window. Default 0.0025")

    # Encoding
    ap.add_argument("--gpu", action="store_true", help="Use NVIDIA NVENC for faster encoding")
    ap.add_argument("--hwaccel", action="store_true", help="Use NVIDIA CUDA decode acceleration (optional)")
    ap.add_argument("--jobs", type=int, default=3, help="Parallel ffmpeg encoding jobs. Default 3")
    ap.add_argument("--concat_copy", action="store_true", help="Concat with -c copy (fastest)")

    ap.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "av1"], help="NVENC codec. Default h264")
    ap.add_argument("--cq", type=int, default=22, help="NVENC CQ (lower=better). Default 22")

    ap.add_argument("--crf", type=int, default=20, help="CPU x264 CRF. Default 20")
    ap.add_argument("--preset", type=str, default="veryfast", help="CPU x264 preset. Default veryfast")
    ap.add_argument("--ab", type=str, default="128k", help="Audio bitrate. Default 128k")
    args = ap.parse_args()

    roi_vals = tuple(float(x.strip()) for x in args.roi.split(","))
    if len(roi_vals) != 4:
        raise ValueError("--roi must be 4 comma-separated floats: x1,y1,x2,y2")

    stream_detect_and_encode(
        video_path=args.input,
        output_path=args.output,
        scan_seconds=args.scan,
        sample_fps=args.sample_fps,
        roi=roi_vals,
        smooth_window=7,
        peak_z=args.peak_z,
        min_gap_sec=args.min_gap,
        pre=args.pre,
        post=args.post,
        merge_gap=args.merge_gap,
        energy_window=args.energy_window,
        energy_z=args.energy_z,
        energy_min=args.energy_min,
        jobs=args.jobs,
        use_gpu=args.gpu,
        use_hwaccel=args.hwaccel,
        gpu_codec=args.vcodec,
        nvenc_cq=args.cq,
        cpu_crf=args.crf,
        cpu_preset=args.preset,
        audio_bitrate=args.ab,
        concat_copy=args.concat_copy,
    )

    print(f"Done ✅ Wrote: {args.output}")


if __name__ == "__main__":
    main()
