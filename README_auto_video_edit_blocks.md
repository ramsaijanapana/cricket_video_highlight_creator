# auto_video_edit_blocks.py

## Description
Automatically extract cricket "shot moments" from a static-camera practice video where the batsman is centered (indoor nets). Outputs a highlight video containing only shots.

Detection: Two complementary detectors (motion-based in ROI):
1) Burst/Peak detector (for full swings)
2) Energy detector (for dead-bat blocks + soft pushes)

Speed: Streaming encode, NVENC, parallel jobs, fast concat.

## Requirements
- ffmpeg + ffprobe in PATH
- pip install opencv-python numpy tqdm

## Usage
Test: python auto_video_edit.py -i "Timeline 1.mov" -o highlights_test.mp4 --scan 120 --gpu --hwaccel --jobs 3 --concat_copy

Full: python auto_video_edit.py -i "Timeline 1.mov" -o highlights_full.mp4 --gpu --hwaccel --jobs 3 --concat_copy

Tuning: --energy_z 1.1 --energy_min 0.002 for blocks