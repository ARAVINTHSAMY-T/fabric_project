from ultralytics import YOLO
import cv2
from preprocess import preprocess_image


def main() -> None:
    # Load the trained model
    model = YOLO("best.pt")

    # Open default webcam (index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index and permissions.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            preprocessed = preprocess_image(frame)

            # Run inference on the current frame
            results = model.predict(source=preprocessed, conf=0.4, verbose=False)

            # Draw predictions on the frame
            annotated = results[0].plot()

            cv2.imshow("Fabric Defect Detection", annotated)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
