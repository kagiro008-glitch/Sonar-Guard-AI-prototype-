from __future__ import annotations

from dataclasses import dataclass, field
import cv2
import numpy as np

from app.config import DEBRIS_CLASSES, WEIGHTS_DIR, YOLO_CONFIDENCE


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float
    label: str
    source: str
    extras: dict = field(default_factory=dict)


class YoloBranch:
    """Known-class detector. Uses Ultralytics if weights exist, else OpenCV blobs."""

    def __init__(self) -> None:
        self._model = None
        weights = WEIGHTS_DIR / "yolo11n.pt"
        sonar_weights = WEIGHTS_DIR / "sonar_yolo11.pt"
        candidate = sonar_weights if sonar_weights.exists() else weights
        if candidate.exists():
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(candidate))
            except Exception:
                self._model = None

    def detect(self, image: np.ndarray) -> list[Detection]:
        if self._model is not None:
            return self._yolo(image)
        return self._blob_fallback(image)

    def _yolo(self, image: np.ndarray) -> list[Detection]:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        results = self._model.predict(rgb, conf=YOLO_CONFIDENCE, verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0]) if box.cls is not None else 0
                name = result.names.get(cls_id, DEBRIS_CLASSES[min(cls_id, len(DEBRIS_CLASSES) - 1)])
                detections.append(
                    Detection(
                        x=x1,
                        y=y1,
                        w=max(x2 - x1, 1),
                        h=max(y2 - y1, 1),
                        score=float(box.conf[0]),
                        label=name,
                        source="yolov11",
                    )
                )
        return detections

    def _blob_fallback(self, image: np.ndarray) -> list[Detection]:
        """Placeholder detector so the demo runs before sonar-trained YOLO weights exist."""
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        h, w = image.shape
        area_total = h * w
        min_area = min(area_total * 0.002, 1500)
        min_area = max(min_area, 180)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < min_area or bw > w * 0.6 or bh > h * 0.6:
                continue
            crop = image[y : y + bh, x : x + bw]
            brightness = float(np.mean(crop) / 255.0)
            aspect = bw / max(bh, 1)
            label = _heuristic_class(aspect, brightness)
            detections.append(
                Detection(
                    x=x,
                    y=y,
                    w=bw,
                    h=bh,
                    score=min(0.45 + brightness * 0.4, 0.92),
                    label=label,
                    source="blob_fallback",
                    extras={"note": "Replace with sonar-trained YOLOv11 weights in weights/sonar_yolo11.pt"},
                )
            )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[:15]


def _heuristic_class(aspect: float, brightness: float) -> str:
    if aspect > 2.4:
        return "plastic_net"
    if 0.75 <= aspect <= 1.25 and brightness > 0.45:
        return "metal_barrel"
    if 0.8 <= aspect <= 1.4:
        return "tire"
    if aspect < 0.7:
        return "wood_wreck"
    return "unclassified"
