from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_ENV = os.getenv("MODEL_PATH", "best1.pt")
MODEL_PATH = MODEL_PATH_ENV if os.path.isabs(MODEL_PATH_ENV) else os.path.join(BASE_DIR, MODEL_PATH_ENV)
CONFIDENCE = float(os.getenv("YOLO_CONF", "0.4"))


class PredictionItem(BaseModel):
    type: str
    confidence: float
    boundingBox: dict[str, float]


class PredictionResponse(BaseModel):
    timestamp: str
    defects: list[PredictionItem]


app = FastAPI(title="Fabric Defect ML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model once at startup so prediction calls stay fast.
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. Set MODEL_PATH to your .pt file path."
        )
    model = YOLO(MODEL_PATH)
except Exception as exc:  # pragma: no cover - surfaced by /health
    model = None
    model_error = str(exc)
else:
    model_error = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": model is not None,
        "modelLoaded": model is not None,
        "cameraConnected": True,
        "modelPath": MODEL_PATH,
        "confidence": CONFIDENCE,
        "error": model_error,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model is not loaded: {model_error}")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty image payload")

    np_image = np.frombuffer(io.BytesIO(payload).getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    height, width = frame.shape[:2]
    results = model.predict(source=frame, conf=CONFIDENCE, verbose=False)

    defects: list[PredictionItem] = []
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            left = max(0.0, min(100.0, (x1 / width) * 100.0))
            top = max(0.0, min(100.0, (y1 / height) * 100.0))
            w = max(0.0, min(100.0, ((x2 - x1) / width) * 100.0))
            h = max(0.0, min(100.0, ((y2 - y1) / height) * 100.0))

            defects.append(
                PredictionItem(
                    type=str(model.names.get(cls_id, f"Class-{cls_id}")),
                    confidence=confidence,
                    boundingBox={"x": left, "y": top, "width": w, "height": h},
                )
            )

    return PredictionResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        defects=defects,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ml_api:app", host="0.0.0.0", port=8000, reload=True)
