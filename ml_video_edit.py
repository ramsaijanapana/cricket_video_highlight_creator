#!/usr/bin/env python3
"""
ml_video_edit.py

Machine learning-based video editor for gameplay highlights.
Trains a CNN model on labeled gameplay frames to predict "keep" vs "skip",
then uses the model to edit videos by keeping high-scoring segments.

Requirements:
- pip install torch torchvision opencv-python numpy tqdm scikit-learn
- ffmpeg + ffprobe in PATH
- GPU recommended (RTX 5080) for training/inference

Usage:
1. Prepare data: Run with --prepare-data to extract frames and labels.
2. Train: python ml_video_edit.py --train --data-dir ./data --epochs 20
3. Edit: python ml_video_edit.py -i input.mp4 -o output.mp4 --model model.pth --threshold 0.7
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# Model Definition
class GameplayCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Assuming 224x224 input -> 56x56 after pools
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# Dataset
class GameplayDataset(Dataset):
    def __init__(self, frames: List[np.ndarray], labels: List[int], transform=None):
        self.frames = frames
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        if self.transform:
            frame = self.transform(frame)
        return frame, torch.tensor(label, dtype=torch.float32)


# Utilities
def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")


def extract_frames(video_path: str, output_dir: str, fps: float = 1.0) -> List[str]:
    """Extract frames from video at given FPS."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path, "-vf", f"fps={fps}",
        "-q:v", "2", f"{output_dir}/frame_%06d.jpg"
    ]
    run(cmd)
    return sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])


def load_labeled_data(data_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """Load frames and labels from data_dir (expects frames/ and labels.csv)."""
    frames_dir = Path(data_dir) / "frames"
    labels_file = Path(data_dir) / "labels.csv"
    frames = []
    labels = []

    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    # Load labels (CSV: frame_name, label)
    label_dict = {}
    with open(labels_file, 'r') as f:
        for line in f:
            name, label = line.strip().split(',')
            label_dict[name] = int(label)

    # Load frames
    for frame_file in sorted(frames_dir.glob("*.jpg")):
        img = cv2.imread(str(frame_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
        labels.append(label_dict.get(frame_file.name, 0))  # Default to 0 if no label

    return frames, labels


def train_model(data_dir: str, epochs: int, model_path: str, batch_size: int = 32):
    """Train the model on labeled data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    frames, labels = load_labeled_data(data_dir)
    dataset = GameplayDataset(frames, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GameplayCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def predict_scores(video_path: str, model_path: str, fps: float = 1.0) -> List[float]:
    """Run model on video frames and return scores."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GameplayCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    scores = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(cap.get(cv2.CAP_PROP_FPS) / fps))

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = transform(frame).unsqueeze(0).to(device)
                score = model(frame_tensor).item()
                scores.append(score)
            frame_count += 1
    cap.release()
    return scores


def edit_video(input_path: str, output_path: str, scores: List[float], threshold: float,
               keep_seconds: float = 2.5, skip_seconds: float = 5.0, fps: float = 1.0):
    """Edit video based on scores: keep segments above threshold."""
    cap = cv2.VideoCapture(input_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Group scores into segments
    segment_length = int(keep_seconds * fps)  # Frames per segment
    keep_segments = []
    i = 0
    while i < len(scores):
        if scores[i] > threshold:
            start = i / fps
            end = min((i + segment_length) / fps, len(scores) / fps)
            keep_segments.append((start, end))
            i += segment_length
        else:
            i += int(skip_seconds * fps)

    # Use ffmpeg to cut and concat
    with tempfile.TemporaryDirectory() as tmpdir:
        parts = []
        for idx, (start, end) in enumerate(keep_segments):
            part_path = Path(tmpdir) / f"part_{idx:05d}.mp4"
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-ss", f"{start}", "-t", f"{end - start}",
                "-c", "copy", str(part_path)
            ]
            run(cmd)
            parts.append(str(part_path))

        # Concat
        list_file = Path(tmpdir) / "list.txt"
        with open(list_file, 'w') as f:
            for p in parts:
                f.write(f"file '{p}'\n")
        cmd_concat = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c", "copy", output_path
        ]
        run(cmd_concat)


def main():
    parser = argparse.ArgumentParser(description="ML-based gameplay video editor")
    parser.add_argument("--prepare-data", action="store_true", help="Extract frames from video for labeling")
    parser.add_argument("--video", help="Video for frame extraction")
    parser.add_argument("--data-dir", default="./data", help="Directory for data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--model", default="gameplay_model.pth", help="Model path")
    parser.add_argument("-i", "--input", help="Input video to edit")
    parser.add_argument("-o", "--output", help="Output video")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for keeping")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS for frame sampling")
    args = parser.parse_args()

    if args.prepare_data:
        if not args.video:
            raise ValueError("--video required for --prepare-data")
        extract_frames(args.video, f"{args.data_dir}/frames", args.fps)
        print(f"Frames extracted to {args.data_dir}/frames. Label them in labels.csv")

    elif args.train:
        train_model(args.data_dir, args.epochs, args.model)

    elif args.input and args.output:
        scores = predict_scores(args.input, args.model, args.fps)
        edit_video(args.input, args.output, scores, args.threshold, fps=args.fps)
        print("Video edited!")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()</content>
<parameter name="filePath">d:\Media\ml_video_edit.py