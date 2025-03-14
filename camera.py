import cv2
import os
import time

class Camera:
    def __init__(self, save_path="recorded_videos/"):
        self.cap = cv2.VideoCapture(0)
        self.recording = False
        self.video_writer = None
        self.save_path = save_path

    def get_frame(self):
        """Capture a frame from the camera."""
        ret, frame = self.cap.read()
        return frame if ret else None

    def start_recording(self):
        """Start recording a video when a person is detected."""
        if not self.recording:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            video_filename = os.path.join(self.save_path, f"video_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
            self.out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
            self.recording = True
            print(f"Recording started: {video_filename}")

    def stop_recording(self):
        """Stop video recording."""
        if self.recording:
            self.out.release()
            print("Recording saved.")
            self.recording = False

    def write_frame(self, frame):
        """Write frame to video file if recording is active."""
        if self.recording:
            self.out.write(frame)

    def release(self):
        """Release the video capture and writer."""
        if self.recording:
            self.stop_recording()
        self.cap.release()
        cv2.destroyAllWindows()
