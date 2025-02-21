from ultralytics import YOLO
import cv2
import time
import numpy as np
import os

class PersonDetector:
    def __init__(self):
        self.model = YOLO('Z:\GitHub\YOLO11\yolo11n.pt')
        self.reference_image = None
        self.reference_features = None
        
        # Initialize SIFT for feature matching
        self.sift = cv2.SIFT_create()
        
        # Create output directory for captured images
        self.output_dir = "captured_references"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def capture_reference(self, frame):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"reference_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        self.reference_image = frame
        # Extract features from reference image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_features = self.sift.detectAndCompute(gray, None)[1]
        return True

    def detect_persons(self, frame, conf=0.5):
        results = self.model(frame, conf=conf)
        annotated_frame = frame.copy()
        
        # Process only if we have reference features
        if self.reference_features is not None:
            # Convert current frame to grayscale and extract features
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_features = self.sift.detectAndCompute(gray, None)[1]
            
            if current_features is not None and len(current_features) > 0:
                # Match features
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(self.reference_features, current_features, k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                match_score = len(good_matches) / len(self.reference_features) if len(self.reference_features) > 0 else 0
                
                # Draw detected persons with match score
                for r in results:
                    for box in r.boxes:
                        if r.names[int(box.cls)] == 'person':
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                        (0, 255, 0) if match_score > 0.1 else (0, 0, 255), 2)
                            cv2.putText(annotated_frame, f"Match: {match_score:.2f}", 
                                      (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 255, 0) if match_score > 0.1 else (0, 0, 255), 2)
        
        return annotated_frame

def main():
    detector = PersonDetector()
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Process frame
        annotated_frame = detector.detect_persons(frame)
        
        # Add instructions
        cv2.putText(annotated_frame, "Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, "Press 'c' to capture reference", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Person Detection", annotated_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if detector.capture_reference(frame):
                print("Reference image captured!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()