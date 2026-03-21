from ultralytics import YOLO

model = YOLO("best1.pt")

# Detect on test images
results = model.predict(
    source="test/images",
    conf=0.4,
    save=True
)

print("Detection Completed")
