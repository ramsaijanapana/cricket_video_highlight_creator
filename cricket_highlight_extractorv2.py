import os
import cv2
import numpy as np
import librosa
import subprocess
import argparse
from scipy.signal import find_peaks, butter, filtfilt

# Attempt to locate FFmpeg (installed automatically with moviepy/imageio)
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = "ffmpeg"  # Fallback to system PATH

class CricketHighlightExtractor:
    def __init__(self, pre_shot_time=1.5, post_shot_time=2.0):
        """
        Initializes the extractor.
        :param pre_shot_time: Seconds to include before the shot is played.
        :param post_shot_time: Seconds to include after the shot is played.
        """
        self.pre_shot_time = pre_shot_time
        self.post_shot_time = post_shot_time

    def _highpass_filter(self, data, cutoff, fs, order=5):
        """
        Applies a high-pass filter to audio data to isolate the high-frequency 
        'crack' of a cricket bat and ignore low-frequency background thuds.
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, data)
        return y

    def detect_shots_audio(self, video_path, cutoff_freq=3000.0, prominence_thresh=3.5):
        """
        Detects cricket shots by finding loud impact spikes in the audio track.
        Uses a high-pass filter and peak detection for accuracy.
        """
        print(f"Analyzing audio for shots in {video_path}...")
        temp_audio_path = "temp_audio_extract.wav"
        
        try:
            # 1. Use raw FFmpeg to extract audio instantly
            print("Extracting audio track for analysis...")
            cmd_extract = [
                FFMPEG_PATH, "-y", "-i", video_path, 
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", temp_audio_path
            ]
            subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if not os.path.exists(temp_audio_path):
                print("Error: Could not extract audio track.")
                return []
                
            # 2. Load audio
            y, sr = librosa.load(temp_audio_path, sr=None)
            
            # 3. Apply High-Pass Filter 
            # Increased to 3000Hz to aggressively ignore net-hits and bowling machine noises
            y_filtered = self._highpass_filter(y, cutoff=cutoff_freq, fs=sr)
            
            # 4. Calculate onset strength on the filtered audio
            # Changed aggregate to np.max to find sharper impulse attacks (the crack)
            onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr, aggregate=np.max)
            
            # 5. Find peaks 
            min_distance_frames = int(5.0 * (sr / 512)) # ~5 seconds apart to prevent overlapping clips
            
            # Prominence increased to filter out footsteps, background clicks, and ball-leaving sounds
            peaks, _ = find_peaks(onset_env, distance=min_distance_frames, prominence=prominence_thresh)
            
            timestamps = librosa.frames_to_time(peaks, sr=sr)
            print(f"Detected {len(timestamps)} shots via Filtered Audio at seconds: {[round(t, 2) for t in timestamps]}")
            return timestamps
            
        except Exception as e:
            print(f"An error occurred during audio detection: {e}")
            return []
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def detect_shots_vision(self, video_path):
        """
        Alternative: Detects shots using visual motion (frame differencing).
        """
        print(f"Analyzing visual motion for shots in {video_path}...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        motion_scores = []
        ret, frame1 = cap.read()
        if not ret: return []
        
        frame1 = cv2.resize(frame1, (640, int(640 * frame1.shape[0] / frame1.shape[1])))
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

        while True:
            ret, frame2 = cap.read()
            if not ret: break
                
            frame2 = cv2.resize(frame2, (640, int(640 * frame2.shape[0] / frame2.shape[1])))
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
            
            frame_delta = cv2.absdiff(gray1, gray2)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            motion = np.sum(thresh)
            motion_scores.append(motion)
            gray1 = gray2
            
        cap.release()

        motion_scores = np.array(motion_scores)
        min_distance_frames = int(5.0 * fps)
        motion_scores = motion_scores / np.max(motion_scores)
        peaks, _ = find_peaks(motion_scores, distance=min_distance_frames, height=0.4)
        
        timestamps = peaks / fps
        print(f"Detected {len(timestamps)} shots via Vision at seconds: {[round(t, 2) for t in timestamps]}")
        return timestamps

    def create_highlights(self, input_video, output_video, timestamps, hw_accel=None, pre_shot_time=None, post_shot_time=None):
        """
        Uses raw FFmpeg subprocesses to extract and concatenate clips.
        This provides maximum speed and zero quality loss.
        """
        print("Generating highlight reel...")
        if len(timestamps) == 0:
            print("No shots found to create highlights.")
            return

        # Override class defaults if specific times are provided
        pre_time = pre_shot_time if pre_shot_time is not None else self.pre_shot_time
        post_time = post_shot_time if post_shot_time is not None else self.post_shot_time

        # Get video duration using ffprobe/cv2
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps
        cap.release()

        temp_clips = []
        list_file = "concat_list.txt"

        try:
            # Step 1: Slice out the individual highlights rapidly
            for i, t in enumerate(timestamps):
                start_t = max(0, t - pre_time)
                clip_duration = min(duration - start_t, pre_time + post_time)
                clip_name = f"temp_highlight_{i}.mp4"
                temp_clips.append(clip_name)

                print(f"Slicing clip {i+1}/{len(timestamps)} ({round(start_t, 2)}s to {round(start_t + clip_duration, 2)}s)...")

                # Quality configuration to match source visually (CRF/CQ 17 is essentially transparent)
                vcodec = "libx264"
                preset = "slow" # Slower preset preserves more quality
                quality_args = ["-crf", "17"] 

                if hw_accel == "nvenc":
                    vcodec = "h264_nvenc"
                    preset = "p6" # High quality preset for NVENC
                    quality_args = ["-cq", "17", "-b:v", "0"] # NVENC constant quality flag
                elif hw_accel == "qsv":
                    vcodec = "h264_qsv"
                    preset = "slow"
                    quality_args = ["-global_quality", "17"] # QSV constant quality flag

                # Frame-accurate slicing (re-encoding with high quality settings to prevent keyframe stuttering)
                cmd_slice = [
                    FFMPEG_PATH, "-y",
                    "-ss", str(start_t),
                    "-i", input_video,
                    "-t", str(clip_duration),
                    "-c:v", vcodec,
                    "-preset", preset
                ] + quality_args + [
                    "-c:a", "aac",
                    "-b:a", "256k", # High quality audio
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt", "yuv420p",
                    clip_name
                ]
                subprocess.run(cmd_slice, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Step 2: Write the clips to a text file for the concat demuxer
            with open(list_file, "w") as f:
                for clip in temp_clips:
                    # FFmpeg requires safe paths in the text file
                    safe_path = os.path.abspath(clip).replace('\\', '/')
                    f.write(f"file '{safe_path}'\n")

            # Step 3: Instantly merge all clips without re-encoding them a second time (-c copy)
            print("Merging clips into final highlight reel...")
            cmd_concat = [
                FFMPEG_PATH, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c", "copy", # Copy streams instantly without quality loss
                output_video
            ]
            subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Highlights successfully saved to {output_video}")

        except Exception as e:
            print(f"Error generating video: {e}")
        finally:
            # Clean up all temporary clip files and the list file
            for clip in temp_clips:
                if os.path.exists(clip):
                    os.remove(clip)
            if os.path.exists(list_file):
                os.remove(list_file)

if __name__ == "__main__":
    # --- Configuration via Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Extract cricket highlights from a video.")
    parser.add_argument("-i", "--input", default="sample_input.mp4", help="Input video file")
    parser.add_argument("-o", "--output", default="automated_sample_input.mp4", help="Output video file")
    parser.add_argument("--hw", default="nvenc", choices=["nvenc", "qsv", "none"], help="Hardware acceleration to use")
    parser.add_argument("--pre", type=float, default=1.5, help="Seconds before the shot (optional)")
    parser.add_argument("--post", type=float, default=2.0, help="Seconds after the shot (optional)")
    parser.add_argument("--cutoff", type=float, default=3000.0, help="Audio highpass cutoff frequency")
    parser.add_argument("--prominence", type=float, default=3.0, help="Peak prominence threshold")
    
    args = parser.parse_args()

    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    HW_ACCELERATION = args.hw if args.hw.lower() != "none" else None
    
    extractor = CricketHighlightExtractor(pre_shot_time=args.pre, post_shot_time=args.post)
    
    # Step 1: Detect Timestamps
    shot_timestamps = extractor.detect_shots_audio(INPUT_FILE, cutoff_freq=args.cutoff, prominence_thresh=args.prominence)
    
    # Step 2: Trim and Merge Video using ultra-fast FFmpeg subprocesses
    if len(shot_timestamps) > 0:
        extractor.create_highlights(
            INPUT_FILE, 
            OUTPUT_FILE, 
            shot_timestamps, 
            hw_accel=HW_ACCELERATION,
            pre_shot_time=args.pre,
            post_shot_time=args.post
        )
    else:
        print("Pipeline finished: No shots were detected.")