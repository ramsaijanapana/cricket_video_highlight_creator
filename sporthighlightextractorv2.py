import os
import cv2
import numpy as np
import subprocess
import math
import librosa
import argparse
from scipy.signal import find_peaks, butter, filtfilt

# Attempt to load AI libraries safely
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import mediapipe as mp
    # Explicitly import the pose module to prevent AttributeError in some environments
    from mediapipe.solutions import pose as mp_pose
except Exception as e:
    print(f"\n[WARNING] MediaPipe failed to initialize: {e}")
    print("The 'mediapipe' model will be unavailable. Please ensure your Python environment supports it.")
    mp = None
    mp_pose = None

# Attempt to locate FFmpeg
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = "ffmpeg"

class SportsHighlightExtractor:
    def __init__(self, pre_shot_time=1.0, post_shot_time=0.5):
        """
        Generic sports highlight extractor supporting multiple AI vision models.
        Timings set tightly to capture just the hit (no long follow-throughs).
        """
        self.pre_shot_time = pre_shot_time
        self.post_shot_time = post_shot_time

    def _highpass_filter(self, data, cutoff, fs, order=5):
        """Isolates high-frequency 'cracks' and ignores low thuds."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

    def detect_shots_audio(self, video_path):
        """Detects loud impact spikes in the audio track."""
        print(f"\n[Audio Module] Analyzing audio track...")
        temp_audio_path = "temp_audio_extract.wav"
        
        try:
            cmd_extract = [
                FFMPEG_PATH, "-y", "-i", video_path, 
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", temp_audio_path
            ]
            subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if not os.path.exists(temp_audio_path): return []
                
            y, sr = librosa.load(temp_audio_path, sr=None)
            y_filtered = self._highpass_filter(y, cutoff=2000.0, fs=sr)
            onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr, aggregate=np.median)
            
            # 1.0s distance allows rapid shots, prominence 1.5 strips out soft taps/drops
            min_distance_frames = int(1.0 * (sr / 512)) 
            peaks, _ = find_peaks(onset_env, distance=min_distance_frames, prominence=1.5)
            timestamps = librosa.frames_to_time(peaks, sr=sr)
            
            print(f"[Audio Module] Found {len(timestamps)} potential acoustic impacts.")
            return timestamps
            
        except Exception as e:
            print(f"Audio detection error: {e}")
            return []
        finally:
            if os.path.exists(temp_audio_path): os.remove(temp_audio_path)

    def _detect_shots_yolo(self, video_path, fps, total_frames):
        """Vision analysis using Ultralytics YOLOv8-Pose."""
        if YOLO is None:
            raise ImportError("YOLO is not installed. Run: pip install ultralytics")
            
        print("\n[Vision Module: YOLOv8] Loading model...")
        model = YOLO("yolov8n-pose.pt")
        cap = cv2.VideoCapture(video_path)
        
        wrist_velocities = []
        prev_wrist = None
        
        print(f"[Vision Module: YOLOv8] Tracking wrists across {total_frames} frames...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            small_frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            results = model.predict(small_frame, verbose=False, classes=[0])
            frame_velocity = 0
            
            if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy[0].cpu().numpy()
                if len(keypoints) >= 11:
                    left_wrist, right_wrist = keypoints[9], keypoints[10]
                    if left_wrist[0] > 0 and right_wrist[0] > 0:
                        curr_wrist = ((left_wrist[0] + right_wrist[0]) / 2, (left_wrist[1] + right_wrist[1]) / 2)
                        if prev_wrist is not None:
                            frame_velocity = math.hypot(curr_wrist[0] - prev_wrist[0], curr_wrist[1] - prev_wrist[1])
                        prev_wrist = curr_wrist
                    else:
                        prev_wrist = None
            
            wrist_velocities.append(frame_velocity)
                
        cap.release()
        return np.array(wrist_velocities)

    def _detect_shots_mediapipe(self, video_path, fps, total_frames):
        """Vision analysis using Google MediaPipe Pose (CPU optimized, handles occlusion better)."""
        if mp is None or mp_pose is None:
            raise ImportError("MediaPipe is not installed correctly or failed to load. Please check the [WARNING] at the top of the console log.")
            
        print("\n[Vision Module: MediaPipe] Loading model...")
        pose_tracker = mp_pose.Pose(
            model_complexity=1, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        cap = cv2.VideoCapture(video_path)
        wrist_velocities = []
        prev_wrist = None
        
        print(f"[Vision Module: MediaPipe] Tracking wrists across {total_frames} frames...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            small_frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            results = pose_tracker.process(frame_rgb)
            frame_velocity = 0
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                if left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5:
                    w, h = small_frame.shape[1], small_frame.shape[0]
                    curr_wrist = (
                        ((left_wrist.x + right_wrist.x) / 2) * w, 
                        ((left_wrist.y + right_wrist.y) / 2) * h
                    )
                    
                    if prev_wrist is not None:
                        frame_velocity = math.hypot(curr_wrist[0] - prev_wrist[0], curr_wrist[1] - prev_wrist[1])
                    prev_wrist = curr_wrist
                else:
                    prev_wrist = None
                    
            wrist_velocities.append(frame_velocity)
            
        cap.release()
        pose_tracker.close()
        return np.array(wrist_velocities)

    def detect_shots_vision(self, video_path, model_choice="mediapipe"):
        """Wrapper to calculate peaks from the chosen vision model."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if model_choice.lower() == "yolo":
            wrist_velocities = self._detect_shots_yolo(video_path, fps, total_frames)
        else:
            wrist_velocities = self._detect_shots_mediapipe(video_path, fps, total_frames)
        
        velocities = cv2.GaussianBlur(wrist_velocities.reshape(-1, 1), (5, 1), 0).flatten()
        if np.max(velocities) > 0:
            velocities = velocities / np.max(velocities)
            
        # Height 0.25 ignores practice taps/shuffles. 
        min_distance_frames = int(1.0 * fps) 
        peaks, _ = find_peaks(velocities, distance=min_distance_frames, height=0.25)
        timestamps = peaks / fps
        
        print(f"[Vision Module] Found {len(timestamps)} potential biomechanical swings.")
        return timestamps

    def detect_shots_multimodal(self, video_path, model_choice="mediapipe", tolerance=0.6):
        """Combines chosen Vision AI and Audio impact detection."""
        print(f"=== Starting Multimodal Detection (Vision: {model_choice.upper()} + Audio) ===")
        
        vision_timestamps = self.detect_shots_vision(video_path, model_choice)
        audio_timestamps = self.detect_shots_audio(video_path)
        
        confirmed_timestamps = []
        print("\n[Fusion Module] Cross-referencing visual and acoustic data...")
        
        # FIX 1: Anchor on VISION (The exact moment of the swing). 
        # This guarantees `post_shot_time` stops recording 0.5s after the bat swings, 
        # preventing the "staring at the bat" effect.
        for v_time in vision_timestamps:
            close_audio = [a_time for a_time in audio_timestamps if abs(a_time - v_time) <= tolerance]
            if close_audio:
                confirmed_timestamps.append(v_time)
                print(f"  [+] CONFIRMED: Swing at {round(v_time, 2)}s matches Impact at {round(close_audio[0], 2)}s")
            else:
                print(f"  [-] REJECTED: Practice swing at {round(v_time, 2)}s (No sound)")
                
        confirmed_timestamps = sorted(list(set(confirmed_timestamps)))
        print(f"\nFinal Confirmed Highlights: {len(confirmed_timestamps)}")
        return confirmed_timestamps

    def create_highlights(self, input_video, output_video, timestamps, hw_accel=None):
        """Extracts and concatenates clips using dynamic window merging to prevent stutter."""
        print("\nGenerating highlight reel...")
        if not timestamps:
            print("No shots found to create highlights.")
            return

        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        
        # --- FIX 2: Dynamic Quality Matching ---
        # Calculate original video bitrate to prevent degradation
        file_size_bytes = os.path.getsize(input_video)
        target_bitrate_bps = (file_size_bytes * 8) / duration
        target_bitrate = f"{int(target_bitrate_bps * 1.2 / 1000)}k" # Add 20% headroom
        print(f"Targeting source quality: ~{target_bitrate} bitrate")
        cap.release()

        # --- FIX 3: Sliding Window Merger ---
        # Instead of dropping rapid-fire shots or causing FFmpeg to stutter,
        # we dynamically merge overlapping highlight windows into continuous clips!
        windows = []
        for t in timestamps:
            start_t = max(0, t - self.pre_shot_time)
            end_t = min(duration, t + self.post_shot_time)
            
            if not windows:
                windows.append([start_t, end_t])
            else:
                last_window = windows[-1]
                if start_t <= last_window[1]: 
                    # They overlap! Extend the previous window smoothly
                    last_window[1] = max(last_window[1], end_t)
                else:
                    windows.append([start_t, end_t])

        temp_clips = []
        list_file = "concat_list.txt"

        try:
            for i, (start_t, end_t) in enumerate(windows):
                clip_duration = end_t - start_t
                clip_name = f"temp_highlight_{i}.mp4"
                temp_clips.append(clip_name)

                print(f"Slicing Clip {i+1}/{len(windows)} ({round(start_t, 2)}s to {round(end_t, 2)}s)...")

                vcodec = "libx264"
                if hw_accel == "nvenc": vcodec = "h264_nvenc"
                elif hw_accel == "qsv": vcodec = "h264_qsv"

                # Swapped to '-preset slow' and explicit bitrate matching for maximum quality
                cmd_slice = [
                    FFMPEG_PATH, "-y", "-ss", str(start_t), "-i", input_video,
                    "-t", str(clip_duration), "-c:v", vcodec, "-preset", "slow",
                    "-b:v", target_bitrate, "-maxrate", target_bitrate, 
                    "-bufsize", f"{int(target_bitrate_bps * 2 / 1000)}k",
                    "-c:a", "aac", "-b:a", "256k",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", clip_name
                ]
                subprocess.run(cmd_slice, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with open(list_file, "w") as f:
                for clip in temp_clips:
                    safe_path = os.path.abspath(clip).replace('\\', '/')
                    f.write(f"file '{safe_path}'\n")

            print("Merging clips into final highlight reel...")
            cmd_concat = [
                FFMPEG_PATH, "-y", "-f", "concat", "-safe", "0",
                "-i", list_file, "-c", "copy", output_video
            ]
            subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ Highlights successfully saved to {output_video}")

        finally:
            for clip in temp_clips:
                if os.path.exists(clip): os.remove(clip)
            if os.path.exists(list_file): os.remove(list_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract sports highlights using Multimodal AI.")
    parser.add_argument("-i", "--input", required=True, help="Path to input video file")
    parser.add_argument("-o", "--output", required=True, help="Path to save output video")
    parser.add_argument("-m", "--model", default="mediapipe", choices=["yolo", "mediapipe"], help="Which AI vision model to use")
    parser.add_argument("--hw_accel", default="nvenc", choices=["nvenc", "qsv", "none"], help="Hardware acceleration")
    
    args = parser.parse_args()
    HW_ACCELERATION = None if args.hw_accel.lower() == "none" else args.hw_accel 
    
    # Strict Hitting Timings: 1.0 second before, 0.5 seconds after
    extractor = SportsHighlightExtractor(pre_shot_time=1.0, post_shot_time=0.5)
    
    # Run pipeline with chosen model
    shot_timestamps = extractor.detect_shots_multimodal(args.input, model_choice=args.model, tolerance=0.6)
    
    if len(shot_timestamps) > 0:
        extractor.create_highlights(args.input, args.output, shot_timestamps, hw_accel=HW_ACCELERATION)
    else:
        print("Pipeline finished: No valid highlights detected.")

# pip install mediapipe ultralytics librosa scipy opencv-python
# python sporthighlightextractorv2.py -i sample_input.mp4 -o output_ai_sample_med.mp4 -m mediapipe
# python sporthighlightextractorv2.py -i sample_input.mp4 -o output_ai_sample_yolo.mp4 -m yolo