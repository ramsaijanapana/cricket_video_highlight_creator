import os
import cv2
import numpy as np
import librosa
from scipy.signal import find_peaks
from moviepy import VideoFileClip, concatenate_videoclips

class CricketHighlightExtractor:
    def __init__(self, pre_shot_time=1.0, post_shot_time=0.5):
        """
        Initializes the extractor.
        :param pre_shot_time: Seconds to include before the shot is played.
        :param post_shot_time: Seconds to include after the shot is played.
        """
        self.pre_shot_time = pre_shot_time
        self.post_shot_time = post_shot_time

    def detect_shots_audio(self, video_path):
        """
        Detects cricket shots by finding loud impact spikes in the audio track.
        This is highly effective for indoor/net sessions.
        """
        print(f"Analyzing audio for shots in {video_path}...")
        temp_audio_path = "temp_audio.wav"
        
        try:
            # Use MoviePy to safely extract audio to a standard WAV file first
            # This avoids librosa's NoBackendError when dealing with MP4s directly
            print("Extracting audio track for analysis...")
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                print("Error: No audio track found in the video.")
                return []
                
            video.audio.write_audiofile(temp_audio_path, logger=None)
            video.close()
            
            # Load the extracted audio using librosa
            y, sr = librosa.load(temp_audio_path, sr=None)
            
            # Calculate the onset strength (sudden changes/impacts in audio)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
            
            # Find peaks in the onset envelope
            # distance: minimum seconds between shots (e.g., a bowler takes at least 5 seconds to bowl again)
            # prominence: threshold for how loud/distinct the sound must be
            min_distance_frames = int(5.0 * (sr / 512)) # ~5 seconds apart
            peaks, _ = find_peaks(onset_env, distance=min_distance_frames, prominence=1.5)
            
            # Convert frame indices to timestamps in seconds
            timestamps = librosa.frames_to_time(peaks, sr=sr)
            
            print(f"Detected {len(timestamps)} shots via Audio at seconds: {[round(t, 2) for t in timestamps]}")
            return timestamps
            
        except Exception as e:
            print(f"An error occurred during audio detection: {e}")
            return []
        finally:
            # Clean up: delete the temporary audio file
            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except OSError:
                    pass

    def detect_shots_vision(self, video_path):
        """
        Alternative: Detects shots using visual motion (frame differencing).
        Optimized by downscaling frames for faster CPU processing.
        """
        print(f"Analyzing visual motion for shots in {video_path}...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        motion_scores = []
        ret, frame1 = cap.read()
        if not ret: return []
        
        # Resize frame to a smaller resolution for much faster processing
        frame1 = cv2.resize(frame1, (640, int(640 * frame1.shape[0] / frame1.shape[1])))
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

        while True:
            ret, frame2 = cap.read()
            if not ret:
                break
                
            frame2 = cv2.resize(frame2, (640, int(640 * frame2.shape[0] / frame2.shape[1])))
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
            
            # Compute absolute difference between current frame and previous frame
            frame_delta = cv2.absdiff(gray1, gray2)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # The sum of white pixels represents the amount of motion
            motion = np.sum(thresh)
            motion_scores.append(motion)
            
            gray1 = gray2
            
        cap.release()

        # Find peaks in motion (the batter swinging)
        motion_scores = np.array(motion_scores)
        min_distance_frames = int(5.0 * fps)
        
        # Normalize motion scores to handle different resolutions
        motion_scores = motion_scores / np.max(motion_scores)
        peaks, _ = find_peaks(motion_scores, distance=min_distance_frames, height=0.4)
        
        timestamps = peaks / fps
        print(f"Detected {len(timestamps)} shots via Vision at seconds: {[round(t, 2) for t in timestamps]}")
        return timestamps

    def create_highlights(self, input_video, output_video, timestamps, hw_accel=None):
        """
        Cuts out the pre-shot and post-shot times and concatenates them.
        Uses multithreading and optional hardware acceleration for speed.
        
        :param hw_accel: 'nvenc' for NVIDIA GPU, 'qsv' for Intel GPU, or None for CPU.
        """
        print("Generating highlight reel...")
        try:
            video = VideoFileClip(input_video)
            duration = video.duration
            
            # Calculate the source video's exact bitrate to match its quality
            file_size_bytes = os.path.getsize(input_video)
            estimated_bitrate_bps = (file_size_bytes * 8) / duration
            source_bitrate = f"{int(estimated_bitrate_bps / 1000)}k"
            print(f"Source video bitrate estimated at {source_bitrate}. Matching output quality...")

            clips = []
            
            for t in timestamps:
                # Calculate start and end times, ensuring they don't go out of video bounds
                start_t = max(0, t - self.pre_shot_time)
                end_t = min(duration, t + self.post_shot_time)
                
                # Extract the subclip using the new method name for MoviePy v2.x
                clip = video.subclipped(start_t, end_t)
                clips.append(clip)
                print(f"Extracted clip: {round(start_t, 2)}s to {round(end_t, 2)}s")
            
            if not clips:
                print("No shots found to create highlights.")
                return

            # Concatenate all highlight clips together
            final_clip = concatenate_videoclips(clips)
            
            # --- FIX: Ensure dimensions are even to prevent vertical video banding ---
            # The H.264 codec requires both width and height to be divisible by 2.
            w, h = final_clip.size
            if w % 2 != 0 or h % 2 != 0:
                new_w = w - (w % 2)
                new_h = h - (h % 2)
                # Crop 1 pixel off to make dimensions even
                final_clip = final_clip.cropped(x1=0, y1=0, x2=new_w, y2=new_h)
            
            # Determine the codec based on requested hardware acceleration
            codec = "libx264" # Default CPU
            if hw_accel == "nvenc":
                codec = "h264_nvenc"
            elif hw_accel == "qsv":
                codec = "h264_qsv"
                
            # Use all available cores for rendering
            num_threads = os.cpu_count() or 4
            
            # Write the result to a file matching the original bitrate
            final_clip.write_videofile(
                output_video, 
                codec=codec, 
                audio_codec="aac", 
                threads=num_threads,
                preset="medium",                     # Medium provides a good balance of speed and size
                bitrate=source_bitrate,              # Matched exactly to the original video
                ffmpeg_params=["-pix_fmt", "yuv420p"] # Guarantee color compatibility
            )
            print(f"Highlights successfully saved to {output_video}")
            
        except Exception as e:
            print(f"Error generating video: {e}")
        finally:
            # Close the video file to release resources
            if 'video' in locals():
                video.close()

if __name__ == "__main__":
    # --- Configuration ---
    INPUT_FILE = "Timeline 1.mov"  # Path to your input cricket video
    OUTPUT_FILE = "Timeline_1_automated_highlights.mp4"
    
    # Enable NVIDIA acceleration if you have an NVIDIA GPU, otherwise set to None
    # Change to "qsv" if you have an Intel integrated GPU
    HW_ACCELERATION = "nvenc" 
    
    # Initialize extractor with your specific timing requirements
    # 1 second before, 0.5 seconds after
    extractor = CricketHighlightExtractor(pre_shot_time=1.0, post_shot_time=0.5)
    
    # Step 1: Detect Timestamps. 
    # Use detect_shots_audio for best results in nets. 
    # Swap to detect_shots_vision if audio is noisy.
    shot_timestamps = extractor.detect_shots_audio(INPUT_FILE)
    
    # Step 2: Trim and Merge Video
    if len(shot_timestamps) > 0:
        extractor.create_highlights(INPUT_FILE, OUTPUT_FILE, shot_timestamps, hw_accel=HW_ACCELERATION)
    else:
        print("Pipeline finished: No shots were detected.")