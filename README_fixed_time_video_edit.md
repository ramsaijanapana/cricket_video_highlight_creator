# fixed_time_video_edit.py

## Description
Keep 2.5s chunks, skip 5s chunks, repeat; output concatenated video (sync-safe).

This script keeps chunks of specified length, skips others, and concatenates them into a new video. Supports parallel processing, GPU encoding (NVENC or Intel QuickSync), and hardware acceleration.

## Requirements
- ffmpeg + ffprobe in PATH
- pip install tqdm (optional for progress bars)

## Usage
Basic usage:
  python fixed_time_video_edit.py -i input.mp4 -o output.mp4

With GPU:
  python fixed_time_video_edit.py -i input.mp4 -o output.mp4 --gpu --gpu-preset p2 --jobs 4

With Intel QuickSync:
  python fixed_time_video_edit.py -i input.mp4 -o output.mp4 --intel --jobs 4

Options:
- --keep: Keep duration (default 2.5s)
- --skip: Skip duration (default 5.0s)
- --jobs: Parallel jobs (default 1)
- --gpu: Use NVENC
- --intel: Use QuickSync
- --hwaccel: Hardware decode acceleration