import pyrealsense2 as rs
import numpy as np
import cv2
import time

class RealSenseCamera:
    def __init__(self):
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable color and depth streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        self.recording = False
        self.out = None

    def start(self):
        """Start the RealSense pipeline"""
        self.pipeline.start(self.config)
        print("RealSense camera started successfully")

    def stop(self):
        """Stop the RealSense pipeline"""
        self.pipeline.stop()
        if self.out:
            self.out.release()
        print("RealSense camera stopped")

    def start_recording(self, output_file):
        """Start recording video to file"""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(output_file, fourcc, 30.0, (640, 480))
        self.recording = True
        print(f"Started recording to {output_file}")

    def stop_recording(self):
        """Stop recording video"""
        if self.recording:
            self.recording = False
            self.out.release()
            print("Recording stopped")

    def get_frame(self):
        """Get a single frame from the camera"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def record_frames(self, duration_seconds=None):
        """Record frames for a specified duration"""
        try:
            start_time = time.time()
            while True:
                color_image, _ = self.get_frame()
                
                if color_image is None:
                    continue

                if self.recording:
                    self.out.write(color_image)

                # Display the frame
                cv2.imshow('RealSense Camera', color_image)

                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Check recording duration
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break

        finally:
            cv2.destroyAllWindows()