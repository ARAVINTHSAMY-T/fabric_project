import argparse
import os

import cv2
from ultralytics import YOLO


def preprocess_image(img: cv2.Mat) -> cv2.Mat:
    # Good lighting is a physical setup requirement; code handles only contrast and filtering.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gaussian = cv2.GaussianBlur(enhanced, (3, 3), 0)
    average = cv2.blur(gaussian, (3, 3))
    boxed = cv2.boxFilter(average, ddepth=-1, ksize=(3, 3), normalize=True)

    sharpen_kernel = (
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)).astype("float32") * -1
    )
    sharpen_kernel[1, 1] = 9.0
    sharpened = cv2.filter2D(boxed, ddepth=-1, kernel=sharpen_kernel)

    median = cv2.medianBlur(sharpened, 3)
    denoised = cv2.bilateralFilter(median, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised


def run_detection(model_path: str, image_path: str, patch_size: int) -> None:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img = preprocess_image(img)
    h, w, _ = img.shape

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = img[y:y + patch_size, x:x + patch_size]

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            results = model(patch)

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(
                        img,
                        (int(x1 + x), int(y1 + y)),
                        (int(x2 + x), int(y2 + y)),
                        (0, 255, 0),
                        2,
                    )

    cv2.imshow("Detected Defects", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam_detection(model_path: str, camera_index: int, conf: float) -> None:
    model = YOLO(model_path)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {camera_index}. Check camera permissions."
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            preprocessed = preprocess_image(frame)
            results = model.predict(source=preprocessed, conf=conf, verbose=False)
            annotated = results[0].plot()

            cv2.imshow("Fabric Defect Detection (Preprocessed)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run defect detection with preprocessing (image or webcam).")
    parser.add_argument(
        "--mode",
        choices=["image", "webcam"],
        default="image",
        help="Run on a single image or realtime webcam frames",
    )
    parser.add_argument(
        "--image",
        default="fabric_large.webp",
        help="Path to the input image",
    )
    parser.add_argument(
        "--model",
        default="best.pt",
        help="Path to YOLO model file",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=640,
        help="Patch size (square tiles)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index for realtime mode",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for detections in webcam mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "webcam":
        run_webcam_detection(args.model, args.camera_index, args.conf)
        return

    run_detection(args.model, args.image, args.patch)


if __name__ == "__main__":
    main()
