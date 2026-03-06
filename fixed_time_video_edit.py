import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# optional progress bars
try:
    from tqdm import tqdm
except ImportError:  # fall back if tqdm isn't installed
    def tqdm(x, **kwargs):
        return x


def run(cmd: list[str]) -> None:
    """Run a command, raise a helpful error if it fails."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n\n"
            f"STDOUT:\n{p.stdout}\n\n"
            f"STDERR:\n{p.stderr}\n"
        )


def run_with_progress(cmd: list[str], duration: float | None = None, desc: str = "") -> None:
    """Run ``ffmpeg`` and show a progress bar parsing stderr.

    If ``duration`` is provided the bar will use it as the total time.  This
    looks for lines like ``time=00:00:10.00`` in ffmpeg stderr and updates the
    bar accordingly.  Falls back to :func:`run` if ``tqdm`` isn’t available or
    parsing fails.
    """
    if 'ffmpeg' not in cmd[0].lower():
        # not an ffmpeg command, just run normally
        return run(cmd)

    # try to open subprocess with stderr pipe
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    bar = None
    if duration is not None:
        bar = tqdm(total=duration, desc=desc or "ffmpeg", unit="s")
    try:
        # read stderr line by line
        while True:
            line = p.stderr.readline()
            if not line:
                break
            if bar is not None:
                # look for time=HH:MM:SS.xxx
                if 'time=' in line:
                    try:
                        tstr = line.split('time=')[1].split(' ')[0]
                        h, m, s = tstr.split(':')
                        secs = float(h) * 3600 + float(m) * 60 + float(s)
                        bar.n = min(secs, duration)
                        bar.refresh()
                    except Exception:
                        pass
        ret = p.wait()
        if bar is not None:
            bar.close()
        if ret != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)} return {ret}")
    except Exception:
        if bar is not None:
            bar.close()
        raise


def ffprobe_duration_seconds(input_path: str) -> float:
    """Get duration in seconds using ffprobe."""
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
    try:
        return float(p.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {p.stdout!r}")


def cut_keep_chunks_and_concat(
    input_path: str,
    output_path: str,
    total_seconds: float,
    keep_seconds: float = 2.5,
    skip_seconds: float = 5.0,
    start_offset: float = 0.0,
    # cpu encoding options
    crf: int = 20,
    preset: str = "veryfast",
    # audio
    audio_bitrate: str = "128k",
    # parallelism / performance
    jobs: int = 1,
    # GPU support (nvenc)
    use_gpu: bool = False,
    gpu_codec: str = "h264",  # h264, hevc, av1
    gpu_preset: str = "p1",
    gpu_cq: int = 19,
    # Intel QuickSync support
    use_intel: bool = False,
    intel_codec: str = "h264",  # h264, hevc
    intel_quality: int = 23,
    # hardware accel flags (cuda or qsv)
    hwaccel: bool = False,
) -> None:
    """
    Keep chunks of length keep_seconds, then skip skip_seconds, repeating.
    Re-encode each kept chunk to fix timestamps/VFR issues, then concat.

    This version supports parallel processing (``jobs``) and optional
    NVIDIA NVENC hardware encoding.  ``use_gpu`` enables ``h264_nvenc``
    (or other nvenc codec specified by ``gpu_codec``) and when
    ``hwaccel`` is True the input decode will use CUDA hardware
    acceleration.  ``jobs`` controls how many ffmpeg processes are
    launched concurrently; set to ``0`` to let ``ThreadPoolExecutor``
    choose.

    ``total_seconds``: how much of the input to process (from start_offset).
    """
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())

    cycle = keep_seconds + skip_seconds
    if total_seconds <= 0:
        raise ValueError("total_seconds must be > 0")

    # choose encoder(s)
    nvenc_map = {"h264": "h264_nvenc", "hevc": "hevc_nvenc", "av1": "av1_nvenc"}
    intel_map = {"h264": "h264_qsv", "hevc": "hevc_qsv"}

    if use_gpu and use_intel:
        raise ValueError("cannot use both --gpu and --intel at the same time")

    if use_gpu:
        encoder = nvenc_map.get(gpu_codec, "h264_nvenc")
    elif use_intel:
        encoder = intel_map.get(intel_codec, "h264_qsv")
    else:
        encoder = "libx264"

    with tempfile.TemporaryDirectory(prefix="keep_chunks_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        parts_dir = tmpdir_path / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)

        # build list of (path, cmd) so we can run them in parallel
        cmds: list[tuple[Path, list[str]]] = []

        # Generate kept chunks
        idx = 0
        t = start_offset
        end_t = start_offset + total_seconds

        while t < end_t:
            part_path = parts_dir / f"part_{idx:05d}.mp4"
            # Ensure we don't cut past the requested range
            clip_len = min(keep_seconds, end_t - t)
            if clip_len <= 0.02:
                break

            # base command
            cmd = ["ffmpeg", "-y"]
            if hwaccel:
                if use_intel:
                    cmd += ["-hwaccel", "qsv"]
                else:
                    cmd += ["-hwaccel", "cuda"]
            cmd += ["-threads", "0"]

            # input & seeking
            cmd += ["-ss", f"{t}", "-t", f"{clip_len}", "-i", input_path]
            cmd += ["-fflags", "+genpts", "-avoid_negative_ts", "make_zero"]

            # encoding options
            cmd += ["-c:v", encoder]
            if use_gpu:
                # use NVENC settings
                cmd += ["-preset", gpu_preset, "-cq", str(gpu_cq), "-rc", "vbr", "-b:v", "0"]
            elif use_intel:
                # basic QSV qual/bitrate mode
                cmd += ["-preset", "veryfast", "-global_quality", str(intel_quality)]
            else:
                cmd += ["-preset", preset, "-crf", str(crf)]

            cmd += ["-c:a", "aac", "-b:a", audio_bitrate, "-movflags", "+faststart", str(part_path)]

            cmds.append((part_path, cmd))
            idx += 1
            t += cycle

        if idx == 0:
            raise RuntimeError("No chunks were created. Check your timings/duration.")

        # execute commands (optionally in parallel)
        if jobs is None or jobs < 1:
            jobs = os.cpu_count() or 1

        if jobs == 1:
            for _, cmd in tqdm(cmds, desc="Encoding chunks", unit="chunk"):
                run(cmd)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                future_to_cmd = {executor.submit(run, cmd): cmd for _, cmd in cmds}
                for future in tqdm(as_completed(future_to_cmd),
                                   total=len(future_to_cmd),
                                   desc="Encoding chunks",
                                   unit="chunk"):
                    future.result()  # will raise if ffmpeg failed


        # Build concat list (ffmpeg concat demuxer)
        list_path = tmpdir_path / "list.txt"
        with list_path.open("w", encoding="utf-8") as f:
            for part_file in sorted(parts_dir.glob("part_*.mp4")):
                # Use absolute path; escape single quotes for safety
                p = str(part_file.resolve()).replace("'", r"'\''")
                f.write(f"file '{p}'\n")

        # Concat (re-encode once more for maximum compatibility)
        # final concat/encode -- we reencode again for maximum compatibility
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
        ]
        if use_gpu:
            cmd_concat += ["-c:v", encoder, "-preset", gpu_preset, "-cq", str(gpu_cq), "-rc", "vbr", "-b:v", "0"]
        elif use_intel:
            cmd_concat += ["-c:v", encoder, "-preset", "veryfast", "-global_quality", str(intel_quality)]
        else:
            cmd_concat += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
        cmd_concat += [
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            output_path,
        ]
        # try to supply duration for progress bar (approximate)
        try:
            run_with_progress(cmd_concat, duration=total_seconds, desc="Final encode")
        except NameError:
            # tqdm not available or helper missing fallback
            run(cmd_concat)


def main():
    parser = argparse.ArgumentParser(
        description="Keep 2.5s chunks, skip 5s chunks, repeat; output concatenated video (sync-safe)."
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file (e.g., input.mp4)")
    parser.add_argument("-o", "--output", required=True, help="Output video file (e.g., output.mp4)")
    parser.add_argument("--duration", type=float, default=None,
                        help="How many seconds of the input to process (default: full duration)")
    parser.add_argument("--sample", type=float, default=None,
                        help="Shortcut: process only the first N seconds (e.g., --sample 60). Overrides --duration.")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset in seconds (default: 0)")
    parser.add_argument("--keep", type=float, default=2.5, help="Keep duration in seconds (default: 2.5)")
    parser.add_argument("--skip", type=float, default=5.0, help="Skip duration in seconds (default: 5.0)")
    parser.add_argument("--crf", type=int, default=20, help="Video quality CRF (lower = better, default 20)")
    parser.add_argument("--preset", default="veryfast", help="x264 preset (default: veryfast)")
    parser.add_argument("--ab", default="128k", help="Audio bitrate (default: 128k)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of parallel ffmpeg jobs (default 1, use 0=auto)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use NVIDIA NVENC encoder instead of x264 (requires RTX series GPU)")
    parser.add_argument("--gpu-codec", choices=["h264", "hevc", "av1"], default="h264",
                        help="NVENC codec to use when --gpu is specified")
    parser.add_argument("--gpu-preset", default="p1", help="NVENC preset (p1..p7, lower is faster) when using --gpu")
    parser.add_argument("--gpu-cq", type=int, default=19,
                        help="NVENC constant quality value (similar to CRF) when using --gpu")
    parser.add_argument("--intel", action="store_true",
                        help="Use Intel QuickSync encoder instead of x264")
    parser.add_argument("--intel-codec", choices=["h264", "hevc"], default="h264",
                        help="QuickSync codec to use when --intel is specified")
    parser.add_argument("--intel-quality", type=int, default=23,
                        help="QuickSync global quality value (lower=better, default 23)")
    parser.add_argument("--hwaccel", action="store_true",
                        help="Enable hardware acceleration for decoding (cuda or qsv) depending on encoder)")
    args = parser.parse_args()

    # Determine duration to process
    if args.sample is not None:
        total_seconds = args.sample
    elif args.duration is not None:
        total_seconds = args.duration
    else:
        full = ffprobe_duration_seconds(args.input)
        total_seconds = max(0.0, full - args.start)

    if total_seconds <= 0:
        print("Nothing to process (duration <= 0).", file=sys.stderr)
        sys.exit(1)

    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Start:   {args.start}s")
    print(f"Process: {total_seconds}s")
    print(f"Pattern: keep {args.keep}s, skip {args.skip}s (cycle {args.keep + args.skip}s)")
    if args.jobs and args.jobs != 1:
        print(f"Parallel jobs: {args.jobs}")
    if args.gpu:
        print("Using GPU encoder (NVENC):", args.gpu_codec)
        print("  preset", args.gpu_preset, "cq", args.gpu_cq)
    if args.intel:
        print("Using Intel QuickSync encoder:", args.intel_codec)
        print("  quality", args.intel_quality)
    if args.hwaccel:
        accel = "qsv" if args.intel else "cuda"
        print("Hardware decode acceleration:", accel)

    cut_keep_chunks_and_concat(
        input_path=args.input,
        output_path=args.output,
        total_seconds=total_seconds,
        keep_seconds=args.keep,
        skip_seconds=args.skip,
        start_offset=args.start,
        crf=args.crf,
        preset=args.preset,
        audio_bitrate=args.ab,
        jobs=args.jobs,
        use_gpu=args.gpu,
        gpu_codec=args.gpu_codec,
        gpu_preset=args.gpu_preset,
        gpu_cq=args.gpu_cq,
        use_intel=args.intel,
        intel_codec=args.intel_codec,
        intel_quality=args.intel_quality,
        hwaccel=args.hwaccel,
    )

    print("Done ✅")


if __name__ == "__main__":
    main()

# Example usage:
# python auto_video_edit.py -i ".\sample_input.mp4" -o highlights_test_v2.mp4 --gpu --hwaccel --jobs 3 --concat_copy --sample_fps 12 --peak_z 1.9 --min_gap 1.5 --min
# _motion_score 0.02 --min_swing_ratio 0.015 --max_burst_sec 1.4 --fdiff_weight  1.0 --roi "0.56,0.12,0.94,0.88"