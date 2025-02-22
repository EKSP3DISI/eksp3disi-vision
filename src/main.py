import cv2
from person_detector import PersonDetector


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
