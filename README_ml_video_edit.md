# ml_video_edit.py

## Description
Machine learning-based video editor for gameplay highlights.
Trains a CNN model on labeled gameplay frames to predict "keep" vs "skip",
then uses the model to edit videos by keeping high-scoring segments.

## Requirements
- pip install torch torchvision opencv-python numpy tqdm scikit-learn
- ffmpeg + ffprobe in PATH
- GPU recommended (RTX 5080) for training/inference

## Usage
1. Prepare data: python ml_video_edit.py --prepare-data --video input.mp4 --data-dir ./data
2. Train: python ml_video_edit.py --train --data-dir ./data --epochs 20
3. Edit: python ml_video_edit.py -i input.mp4 -o output.mp4 --model model.pth --threshold 0.7