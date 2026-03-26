from ultralytics import YOLO
import cv2
import argparse
from preprocess import preprocess_image, preprocess_image_light


def _try_open_camera(index: int, backend: int) -> cv2.VideoCapture | None:
    """
    Open a camera and verify it can produce frames.
    """
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None

    # Some backends report opened but fail on read; verify one frame.
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def _available_backends() -> list[tuple[int, str]]:
    return [
        (cv2.CAP_DSHOW, "dshow"),
        (cv2.CAP_MSMF, "msmf"),
        (cv2.CAP_ANY, "any"),
    ]


def find_working_camera(max_index=10, forced_backend: str | None = None):
    """
    Automatically find the first working camera index/backend pair.
    """
    backends = _available_backends()

    if forced_backend is not None:
        selected = [item for item in backends if item[1] == forced_backend]
        if not selected:
            raise ValueError(f"Unsupported backend '{forced_backend}'.")
        backends = selected

    for i in range(max_index):
        for backend, backend_name in backends:
            cap = _try_open_camera(i, backend)
            if cap is not None:
                cap.release()
                print(f"[INFO] Using camera index: {i} (backend={backend_name.upper()})")
                return i, backend, backend_name.upper()

    raise RuntimeError(
        "No working camera found. Check camera permissions, USB connection, and whether another app is using the webcam."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time fabric defect detection from webcam.")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index to use (e.g., 0, 1, 2). If omitted, auto-detect is used.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "dshow", "msmf", "any"],
        help="Video backend to use. 'auto' tries multiple backends.",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=20,
        help="Maximum camera index to scan in auto-detect mode (exclusive).",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="none",
        choices=["none", "light", "full"],
        help="Image preprocessing: 'none' (fastest), 'light' (balanced), 'full' (slowest but best quality).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load trained YOLO model
    model = YOLO("best.pt")
    print(f"[INFO] Model classes: {model.names}")

    # Auto-detect working camera (USB or built-in) unless manually set.
    forced_backend = None if args.backend == "auto" else args.backend

    if args.camera is None:
        cam_index, cam_backend, backend_name = find_working_camera(
            max_index=args.max_index,
            forced_backend=forced_backend,
        )
        cap = cv2.VideoCapture(cam_index, cam_backend)
        print(f"[INFO] Opening camera index {cam_index} with backend={backend_name}")
    else:
        cam_index = args.camera
        backend_map = {name: code for code, name in _available_backends()}
        selected_backend_name = "any" if args.backend == "auto" else args.backend
        selected_backend_code = backend_map[selected_backend_name]
        cap = cv2.VideoCapture(cam_index, selected_backend_code)
        print(
            f"[INFO] Opening manually selected camera index {cam_index} with backend={selected_backend_name.upper()}"
        )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {cam_index}. Try another index: python webcam_detect.py --camera 1"
        )

    # Optimize camera settings for better quality and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[INFO] Camera started successfully.")
    print(f"[INFO] Preprocessing mode: {args.preprocess.upper()}")
    print("[INFO] Press 'q' to quit.")

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame.")
                break

            frame_count += 1
            
            # Apply preprocessing based on argument
            if args.preprocess == "full":
                processed_frame = preprocess_image(frame)
            elif args.preprocess == "light":
                processed_frame = preprocess_image_light(frame)
            else:  # "none"
                processed_frame = frame
            
            # Run inference
            results = model(processed_frame, conf=0.4)

            # Draw detection results
            annotated_frame = results[0].plot()
            
            # Resize for better display (optional)
            display_frame = cv2.resize(annotated_frame, (960, 540))

            # Show output
            cv2.imshow("Fabric Defect Detection", display_frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera closed.")


if __name__ == "__main__":
    main()