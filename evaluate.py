from pathlib import Path
import shutil

from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Generate confusion matrices per split
splits = ["train", "val", "test"]
output_base = Path("runs/confusion_matrices")

for split in splits:
	metrics = model.val(data="data.yaml", split=split, plots=True)

	print(f"[{split}] Precision:", metrics.box.mp)
	print(f"[{split}] Recall:", metrics.box.mr)
	print(f"[{split}] mAP50:", metrics.box.map50)
	print(f"[{split}] mAP50-95:", metrics.box.map)

	save_dir = Path(metrics.save_dir)
	target_dir = output_base / split
	target_dir.mkdir(parents=True, exist_ok=True)

	for name in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
		src = save_dir / name
		if src.exists():
			shutil.copy2(src, target_dir / name)
