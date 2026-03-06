# cricket_highlight_extractor.py

## Description
A class for extracting cricket highlights by detecting shots via audio (impact spikes) or vision (motion detection).

## Requirements
- pip install opencv-python numpy librosa scipy moviepy

## Usage
from cricket_highlight_extractor import CricketHighlightExtractor

extractor = CricketHighlightExtractor()
timestamps = extractor.detect_shots_audio('video.mp4')
# or
timestamps = extractor.detect_shots_vision('video.mp4')
# Then use moviepy to clip highlights