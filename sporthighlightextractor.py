import os
import cv2
import numpy as np
import subprocess
import math
import librosa
import argparse
from scipy.signal import find_peaks, butter, filtfilt

# You will need to install ultralytics: `pip install ultralytics`
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics using: pip install ultralytics")
    exit()

# Attempt to locate FFmpeg
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = "ffmpeg"

class SportsHighlightExtractor:
    def __init__(self, pre_shot_time=1.5, post_shot_time=1.0):
        """
        Generic sports highlight extractor for bat/club/racket sports.
        """
        # Slightly longer windows capture the full windup and follow-through
        self.pre_shot_time = pre_shot_time
        self.post_shot_time = post_shot_time

    def _highpass_filter(self, data, cutoff, fs, order=5):
        """
        Applies a high-pass filter to audio data to isolate the high-frequency 
        'crack' of a bat and ignore low-frequency background noise.
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, data)
        return y

    def detect_shots_audio(self, video_path):
        """
        Detects loud impact spikes in the audio track.
        """
        print(f"\n[Audio Module] Analyzing audio track...")
        temp_audio_path = "temp_audio_extract.wav"
        
        try:
            cmd_extract = [
                FFMPEG_PATH, "-y", "-i", video_path, 
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", temp_audio_path
            ]
            subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if not os.path.exists(temp_audio_path):
                return []
                
            y, sr = librosa.load(temp_audio_path, sr=None)
            y_filtered = self._highpass_filter(y, cutoff=2000.0, fs=sr)
            onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr, aggregate=np.median)
            
            min_distance_frames = int(3.0 * (sr / 512)) # ~3 seconds apart
            peaks, _ = find_peaks(onset_env, distance=min_distance_frames, prominence=1.5)
            timestamps = librosa.frames_to_time(peaks, sr=sr)
            
            print(f"[Audio Module] Found {len(timestamps)} potential acoustic impacts.")
            return timestamps
            
        except Exception as e:
            print(f"Audio detection error: {e}")
            return []
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def detect_shots_ai_pose(self, video_path):
        """
        Uses YOLOv8 to track the biomechanics of the player's wrists.
        A sudden spike in wrist velocity indicates a swing.
        """
        print("\n[Vision Module] Loading YOLOv8 Pose Model...")
        model = YOLO("yolov8n-pose.pt")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        wrist_velocities = []
        prev_wrist = None
        
        print(f"[Vision Module] Analyzing biomechanics across {total_frames} frames...")
        
        frame_idx = 0
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
            frame_idx += 1
                
        cap.release()
        
        velocities = np.array(wrist_velocities)
        velocities = cv2.GaussianBlur(velocities.reshape(-1, 1), (5, 1), 0).flatten()
        
        min_distance_frames = int(3.0 * fps)
        if np.max(velocities) > 0:
            velocities = velocities / np.max(velocities)
            
        peaks, _ = find_peaks(velocities, distance=min_distance_frames, height=0.4)
        timestamps = peaks / fps
        
        print(f"[Vision Module] Found {len(timestamps)} potential biomechanical swings.")
        return timestamps

    def detect_shots_multimodal(self, video_path, tolerance=0.6):
        """
        Combines AI Biomechanics (Vision) and Audio impact detection.
        A shot is only confirmed if a swing is visually detected AND 
        a loud bat/club impact is heard within 'tolerance' seconds.
        """
        print("=== Starting Multimodal Detection (Vision + Audio) ===")
        vision_timestamps = self.detect_shots_ai_pose(video_path)
        audio_timestamps = self.detect_shots_audio(video_path)
        
        confirmed_timestamps = []
        
        print("\n[Fusion Module] Cross-referencing visual and acoustic data...")
        # Check visual swings against audio peaks
        for v_time in vision_timestamps:
            close_audio = [a_time for a_time in audio_timestamps if abs(a_time - v_time) <= tolerance]
            
            if close_audio:
                # The audio peak is usually the exact millisecond of impact. 
                impact_time = close_audio[0]
                confirmed_timestamps.append(impact_time)
                print(f"  [+] CONFIRMED: Swing at {round(v_time, 2)}s matches Impact at {round(impact_time, 2)}s")
            else:
                print(f"  [-] REJECTED: Practice swing at {round(v_time, 2)}s (No acoustic impact detected)")
                
        # Handle cases where audio heard a crack, but no swing was seen
        for a_time in audio_timestamps:
            close_vision = [v_time for v_time in vision_timestamps if abs(v_time - a_time) <= tolerance]
            if not close_vision:
                print(f"  [-] REJECTED: Loud noise at {round(a_time, 2)}s (No visual swing detected)")
                
        print(f"\nFinal Confirmed Multimodal Highlights: {len(confirmed_timestamps)}")
        return confirmed_timestamps

    def create_highlights(self, input_video, output_video, timestamps, hw_accel=None):
        """
        Uses raw FFmpeg subprocesses to extract and concatenate clips instantly.
        """
        print("\nGenerating highlight reel...")
        if not timestamps:
            print("No shots found to create highlights.")
            return

        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps
        cap.release()

        temp_clips = []
        list_file = "concat_list.txt"

        try:
            for i, t in enumerate(timestamps):
                start_t = max(0, t - self.pre_shot_time)
                clip_duration = min(duration - start_t, self.pre_shot_time + self.post_shot_time)
                clip_name = f"temp_highlight_{i}.mp4"
                temp_clips.append(clip_name)

                print(f"Slicing clip {i+1}/{len(timestamps)} ({round(start_t, 2)}s to {round(start_t + clip_duration, 2)}s)...")

                vcodec = "libx264"
                if hw_accel == "nvenc":
                    vcodec = "h264_nvenc"
                elif hw_accel == "qsv":
                    vcodec = "h264_qsv"

                cmd_slice = [
                    FFMPEG_PATH, "-y",
                    "-ss", str(start_t),
                    "-i", input_video,
                    "-t", str(clip_duration),
                    "-c:v", vcodec,
                    "-preset", "fast",
                    "-crf", "18", 
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt", "yuv420p",
                    clip_name
                ]
                subprocess.run(cmd_slice, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with open(list_file, "w") as f:
                for clip in temp_clips:
                    safe_path = os.path.abspath(clip).replace('\\', '/')
                    f.write(f"file '{safe_path}'\n")

            print("Merging clips into final highlight reel...")
            cmd_concat = [
                FFMPEG_PATH, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                output_video
            ]
            subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Highlights successfully saved to {output_video}")

        except Exception as e:
            print(f"Error generating video: {e}")
        finally:
            for clip in temp_clips:
                if os.path.exists(clip):
                    os.remove(clip)
            if os.path.exists(list_file):
                os.remove(list_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract sports highlights using AI Biomechanics and Audio.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input video file")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output highlight reel (e.g., output.mp4)")
    parser.add_argument("--hw_accel", default="nvenc", choices=["nvenc", "qsv", "none"], help="Hardware acceleration to use (nvenc, qsv, or none)")
    
    args = parser.parse_args()
    
    # Options: "nvenc" (NVIDIA), "qsv" (Intel), or None (CPU)
    HW_ACCELERATION = None if args.hw_accel.lower() == "none" else args.hw_accel 
    
    # 1.5 seconds before swing, 1.0 second after to catch the full baseball/cricket follow-through
    extractor = SportsHighlightExtractor(pre_shot_time=1.5, post_shot_time=1.0)
    
    # Step 1: Detect Timestamps using Multimodal Data (AI Biomechanics + Filtered Audio)
    shot_timestamps = extractor.detect_shots_multimodal(args.input, tolerance=0.6)
    
    # Step 2: Trim and Merge Video using ultra-fast FFmpeg subprocesses
    if len(shot_timestamps) > 0:
        extractor.create_highlights(args.input, args.output, shot_timestamps, hw_accel=HW_ACCELERATION)
    else:
        print("Pipeline finished: No valid multimodal swings were detected.")
    
#python sports_highlights.py -i sample_input.mp4 -o my_new_highlights.mp4

# **Advanced Usage (specifying no hardware acceleration):**
# ```bash
# python sports_highlights.py -i "C:/Videos/game_footage.mp4" -o "C:/Videos/Highlights/game_highlights.mp4" --hw_accel none