from ultralytics import YOLO

# Load trained model
model = YOLO("best1.pt")

# Predict on another dataset folder
results = model.predict(
    source="new_dataset/images",   # your new dataset path
    conf=0.4,
    save=True
)

print("Prediction Completed")
