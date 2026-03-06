# auto_video_edit.py

## Description
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

## Requirements
- ffmpeg + ffprobe in PATH
- pip install opencv-python numpy tqdm

## Usage
Recommended usage (your RTX 5080):
  python auto_video_edit.py -i "Timeline 1.mov" -o highlights_test.mp4 --scan 120 --gpu --hwaccel --jobs 3 --concat_copy
  python auto_video_edit.py -i "Timeline 1.mov" -o highlights_full.mp4 --gpu --hwaccel --jobs 3 --concat_copy

Tuning:
- Faster scan: --sample_fps 6 (or 4)
- More sensitive: --peak_z 2.0 (or 1.7)
- Fewer tiny clips: --merge_gap 0.8