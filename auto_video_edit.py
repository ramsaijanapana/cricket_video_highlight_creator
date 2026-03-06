#!/usr/bin/env python3
"""
auto_video_edit.py (streaming encode)

Cricket-net highlight extractor for a static camera + centered batsman.

NEW: Starts encoding WHILE scanning (producer/consumer pipeline)
- Scan frames (CPU) -> detect motion bursts (shot events) -> build clip ranges
- Finalize/merge ranges as scan progresses -> enqueue ffmpeg encodes immediately
- Concat encoded parts at the end

Speed features:
- NVIDIA NVENC encoding (--gpu) with codec (--vcodec) and quality (--cq)
- Optional CUDA decode acceleration (--hwaccel)
- Parallel segment encoding (--jobs)
- Optional fast concat without re-encode (--concat_copy)

Requirements:
- ffmpeg + ffprobe in PATH
- pip install opencv-python numpy tqdm

Recommended usage (your RTX 5080):
  python auto_video_edit.py -i "Timeline 1.mov" -o highlights_test.mp4 --scan 120 --gpu --hwaccel --jobs 3 --concat_copy
  python auto_video_edit.py -i "Timeline 1.mov" -o highlights_full.mp4 --gpu --hwaccel --jobs 3 --concat_copy

Tuning:
- Faster scan: --sample_fps 6 (or 4)
- More sensitive: --peak_z 2.0 (or 1.7)
- Fewer tiny clips: --merge_gap 0.8
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


def ffprobe_duration_seconds(input_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{p.stderr}")
    return float(p.stdout.strip())


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
    # Choose NVENC encoder
    nvenc_map = {"h264": "h264_nvenc", "hevc": "hevc_nvenc", "av1": "av1_nvenc"}
    if gpu_codec not in nvenc_map:
        raise ValueError("gpu_codec must be one of: h264, hevc, av1")
    nvenc_encoder = nvenc_map[gpu_codec]

    cmd = ["ffmpeg", "-y"]
    if use_hwaccel:
        cmd += ["-hwaccel", "cuda"]

    # Accurate seek: -i first, then -ss
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
    min_peak_gap_sec: float,
    pre: float,
    post: float,
    merge_gap: float,
    jobs: int,
    use_gpu: bool,
    use_hwaccel: bool,
    gpu_codec: str,
    nvenc_cq: int,
    cpu_crf: int,
    cpu_preset: str,
    audio_bitrate: str,
    concat_copy: bool,
    warmup_sec: float,
    min_motion_score: float,
    min_swing_ratio: float,
    max_burst_sec: float,
    fdiff_weight: float,
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
    duration = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)

    if scan_seconds is None:
        scan_frames_limit = total_frames
    else:
        scan_frames_limit = min(total_frames, int(scan_seconds * native_fps))

    # Background subtractor (static camera)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

    # Rolling stats for robust z-score (median + MAD)
    # Keep ~6 seconds of samples, at least 30.
    hist_len = max(30, int(effective_scan_fps * 6))
    hist = deque(maxlen=hist_len)

    # Burst detection state
    in_burst = False
    burst_max_z = -1e9
    burst_max_t = 0.0
    burst_max_score = 0.0
    burst_max_swing = 0.0
    burst_start_t = 0.0
    prev_gray = None

    last_event_t = -1e9

    # Pending merged range (start,end) not yet submitted for encoding
    pending: list[float] | None = None
    pending_idx = 0

    # Encoding executor
    workers = max(1, int(jobs))

    # We'll store futures -> idx, and idx -> part file path
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

        # ThreadPoolExecutor: good enough; we mostly spawn external ffmpeg processes
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

                # Current time
                t = frame_i / native_fps

                # If pending range is "safe" (we are past its end + merge_gap), finalize it and submit
                if pending is not None and t > (pending[1] + merge_gap):
                    idx = pending_idx
                    s, e = pending
                    futures.append(ex.submit(submit_range, s, e, idx))
                    pending_idx += 1
                    pending = None

                # Compute motion score in ROI
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = roi
                X1, Y1, X2, Y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                crop = frame[Y1:Y2, X1:X2]

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                fg = fgbg.apply(gray)
                fg = cv2.medianBlur(fg, 5)
                _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

                # Cricket-specific motion features:
                # - swing area: upper part where bat/arms move
                # - lower area: legs/body translation (penalize to avoid walk/stance noise)
                hh, ww = fg.shape[:2]
                split = max(1, int(hh * 0.58))
                fg_upper = fg[:split, :]
                fg_lower = fg[split:, :]

                full_ratio = float(np.count_nonzero(fg)) / float(fg.size)
                upper_ratio = float(np.count_nonzero(fg_upper)) / float(max(1, fg_upper.size))
                lower_ratio = float(np.count_nonzero(fg_lower)) / float(max(1, fg_lower.size))

                # Emphasize upper-body/bat movement, suppress lower-body movement
                score_mog = (0.70 * upper_ratio) + (0.35 * full_ratio) - (0.40 * lower_ratio)
                if score_mog < 0.0:
                    score_mog = 0.0

                # Frame-difference motion (helps when background subtractor under/over-adapts)
                if prev_gray is None:
                    fdiff = np.zeros_like(gray)
                else:
                    fdiff = cv2.absdiff(gray, prev_gray)
                _, fdiff = cv2.threshold(fdiff, 18, 255, cv2.THRESH_BINARY)
                fdiff = cv2.medianBlur(fdiff, 3)
                prev_gray = gray

                fd_full = float(np.count_nonzero(fdiff)) / float(max(1, fdiff.size))
                fd_upper = float(np.count_nonzero(fdiff[:split, :])) / float(max(1, fdiff[:split, :].size))

                # Contact zone near batsman (right-mid of ROI) to favor actual play interaction
                ch, cw = fdiff.shape[:2]
                cx1, cx2 = int(cw * 0.58), int(cw * 0.98)
                cy1, cy2 = int(ch * 0.30), int(ch * 0.85)
                contact = fdiff[cy1:cy2, cx1:cx2]
                contact_ratio = float(np.count_nonzero(contact)) / float(max(1, contact.size))

                score_fd = (0.55 * fd_upper) + (0.25 * fd_full) + (0.90 * contact_ratio)
                score = score_mog + (fdiff_weight * score_fd)
                if score < 0.0:
                    score = 0.0

                # Update rolling stats
                hist.append(score)
                if len(hist) >= max(10, smooth_window):
                    arr = np.array(hist, dtype=np.float32)
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

                    # Burst logic (hysteresis): enter when z > peak_z, exit when z drops below peak_z*0.6
                    enter_thr = peak_z
                    exit_thr = peak_z * 0.6

                    if not in_burst:
                        if z > enter_thr and (t - last_event_t) >= min_peak_gap_sec and t >= warmup_sec:
                            in_burst = True
                            burst_start_t = t
                            burst_max_z = z
                            burst_max_t = t
                            burst_max_score = score
                            burst_max_swing = upper_ratio
                    else:
                        if z > burst_max_z:
                            burst_max_z = z
                            burst_max_t = t
                        if score > burst_max_score:
                            burst_max_score = score
                        if upper_ratio > burst_max_swing:
                            burst_max_swing = upper_ratio
                        if z < exit_thr:
                            # Burst ended -> finalize event time at max within burst
                            burst_dur = t - burst_start_t
                            event_t = burst_max_t
                            in_burst = False

                            # Reject slow/weak/non-swing bursts (common false positives)
                            if (
                                burst_dur <= max_burst_sec
                                and burst_max_score >= min_motion_score
                                and burst_max_swing >= min_swing_ratio
                            ):
                                last_event_t = event_t

                                # Build candidate keep range
                                rs = event_t - pre
                                re = event_t + post

                                # Merge into pending or flush pending
                                if pending is None:
                                    pending = [rs, re]
                                else:
                                    if rs <= pending[1] + merge_gap:
                                        pending[1] = max(pending[1], re)
                                    else:
                                        # Submit existing pending immediately (we know next segment is separated)
                                        idx = pending_idx
                                        s, e = pending
                                        futures.append(ex.submit(submit_range, s, e, idx))
                                        pending_idx += 1
                                        pending = [rs, re]

                frame_i += 1
                pbar.update(1)

            pbar.close()
            cap.release()

            # If a burst is still open at EOF, finalize it.
            if in_burst:
                end_t = frame_i / max(native_fps, 1e-6)
                burst_dur = end_t - burst_start_t
                event_t = burst_max_t
                in_burst = False
                if (
                    burst_dur <= max_burst_sec
                    and burst_max_score >= min_motion_score
                    and burst_max_swing >= min_swing_ratio
                ):
                    last_event_t = event_t
                    rs = event_t - pre
                    re = event_t + post
                    if pending is None:
                        pending = [rs, re]
                    else:
                        if rs <= pending[1] + merge_gap:
                            pending[1] = max(pending[1], re)
                        else:
                            idx = pending_idx
                            s, e = pending
                            futures.append(ex.submit(submit_range, s, e, idx))
                            pending_idx += 1
                            pending = [rs, re]

            # Flush any pending range
            if pending is not None:
                idx = pending_idx
                s, e = pending
                futures.append(ex.submit(submit_range, s, e, idx))
                pending_idx += 1
                pending = None

            # Collect results (ensures all ffmpeg processes finished)
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Finishing encodes", unit="clip"):
                idx, part = fut.result()
                part_paths[idx] = part

        if not part_paths:
            raise RuntimeError("No clips were produced. Try lowering --peak_z or expanding --roi.")

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
            # Safer: final encode
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
    ap = argparse.ArgumentParser(description="Keep only cricket shots and export highlights (streaming encode).")
    ap.add_argument("-i", "--input", required=True, help='Input video, e.g. "Timeline 1.mov"')
    ap.add_argument("-o", "--output", required=True, help="Output video, e.g. highlights.mp4")

    ap.add_argument("--scan", type=float, default=None,
                    help="Scan only first N seconds (e.g. 120). Default = full video.")
    ap.add_argument("--sample_fps", type=float, default=12.0,
                    help="Analysis FPS (lower=faster scan). Default 12.")
    ap.add_argument("--peak_z", type=float, default=2.5,
                    help="Sensitivity: lower = more shots detected. Default 2.5")
    ap.add_argument("--min_gap", type=float, default=1.8,
                    help="Minimum seconds between detected shots. Default 1.8")
    ap.add_argument("--warmup_sec", type=float, default=2.0,
                    help="Ignore detections in first N seconds while background model stabilizes.")

    ap.add_argument("--pre", type=float, default=1.8,
                    help="Seconds to keep BEFORE shot. Default 0.8")
    ap.add_argument("--post", type=float, default=0.8,
                    help="Seconds to keep AFTER shot. Default 1.4")
    ap.add_argument("--merge_gap", type=float, default=0.20,
                    help="Merge segments if gap <= this. Default 0.20")

    ap.add_argument("--roi", type=str, default="0.55,0.10,0.92,0.85",
                    help="ROI normalized x1,y1,x2,y2. Default tuned for centered batsman.")
    ap.add_argument("--min_motion_score", type=float, default=0.035,
                    help="Minimum peak motion score to accept event. Increase to reduce false positives.")
    ap.add_argument("--min_swing_ratio", type=float, default=0.028,
                    help="Minimum upper-ROI motion ratio for event acceptance.")
    ap.add_argument("--max_burst_sec", type=float, default=1.10,
                    help="Reject motion bursts longer than this (walking/stance changes).")
    ap.add_argument("--fdiff_weight", type=float, default=0.70,
                    help="Weight of frame-diff motion added to detector score. Increase if shots are missed.")

    # Speed / encoding
    ap.add_argument("--gpu", action="store_true", help="Use NVIDIA NVENC for faster encoding")
    ap.add_argument("--hwaccel", action="store_true", help="Use NVIDIA decode acceleration (cuda). Optional.")
    ap.add_argument("--jobs", type=int, default=3, help="Parallel ffmpeg segment jobs (default 3)")
    ap.add_argument("--concat_copy", action="store_true", help="Concat with -c copy (no re-encode)")
    ap.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "av1"],
                    help="GPU codec if --gpu is set (default h264)")
    ap.add_argument("--cq", type=int, default=22,
                    help="NVENC constant quality (lower=better). Default 22")

    ap.add_argument("--crf", type=int, default=20, help="CPU x264 quality CRF (lower=better). Default 20")
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
        min_peak_gap_sec=args.min_gap,
        pre=args.pre,
        post=args.post,
        merge_gap=args.merge_gap,
        jobs=args.jobs,
        use_gpu=args.gpu,
        use_hwaccel=args.hwaccel,
        gpu_codec=args.vcodec,
        nvenc_cq=args.cq,
        cpu_crf=args.crf,
        cpu_preset=args.preset,
        audio_bitrate=args.ab,
        concat_copy=args.concat_copy,
        warmup_sec=args.warmup_sec,
        min_motion_score=args.min_motion_score,
        min_swing_ratio=args.min_swing_ratio,
        max_burst_sec=args.max_burst_sec,
        fdiff_weight=args.fdiff_weight,
    )

    print(f"Done ✅ Wrote: {args.output}")


if __name__ == "__main__":
    main()
