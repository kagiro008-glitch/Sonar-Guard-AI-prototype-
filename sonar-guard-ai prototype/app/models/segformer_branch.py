from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import DEBRIS_CLASSES


@dataclass
class SegMask:
    label: str
    mask: np.ndarray
    confidence_mask: float
    bbox: tuple[int, int, int, int]
    extras: dict = field(default_factory=dict)


_SEABED_LABEL = "seabed"
_SHADOW_LABEL = "shadow"
_ALL_CLASSES = DEBRIS_CLASSES + [_SEABED_LABEL, _SHADOW_LABEL]


class SegFormerBranch:
    def __init__(self) -> None:
        self._hf_model = None
        self._hf_processor = None
        self._backend = "fallback"
        self._warned = False
        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

            model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
            try:
                processor = AutoImageProcessor.from_pretrained(model_name)
                model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
                model.eval()
                self._hf_processor = processor
                self._hf_model = model
                self._backend = "transformers"
            except Exception:
                if not self._warned:
                    warnings.warn(
                        "SegFormer real weights unavailable; using OpenCV texture fallback. "
                        "Install transformers>=4.40 and a sonar fine-tuned checkpoint in weights/ to enable.",
                        stacklevel=2,
                    )
                    self._warned = True
        except Exception:
            if not self._warned:
                warnings.warn(
                    "transformers package not installed; SegFormer using OpenCV texture fallback.",
                    stacklevel=2,
                )
                self._warned = True

    def detect(self, image: np.ndarray) -> list[SegMask]:
        if self._backend == "transformers" and self._hf_model is not None:
            return self._transformers_detect(image)
        return self._fallback_detect(image)

    def _transformers_detect(self, image: np.ndarray) -> list[SegMask]:
        try:
            import torch

            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            inputs = self._hf_processor(images=rgb, return_tensors="pt")
            with torch.no_grad():
                outputs = self._hf_model(**inputs)
            logits = outputs.logits
            upsampled = torch.nn.functional.interpolate(
                logits, size=image.shape, mode="bilinear", align_corners=False
            )
            pred = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
            h, w = image.shape
            ade_to_sonar = {
                17: "plastic_net",
                31: "metal_barrel",
                71: "tire",
                13: "wood_wreck",
                29: "anomaly",
            }
            masks: list[SegMask] = []
            for ade_idx, label in ade_to_sonar.items():
                binary = (pred == ade_idx).astype(np.uint8)
                if binary.sum() < 64:
                    continue
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    x, y, bw, bh = cv2.boundingRect(c)
                    if bw * bh < 64:
                        continue
                    crop = binary[y : y + bh, x : x + bw]
                    conf = float(np.count_nonzero(crop) / max(1, crop.size))
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    full_mask[y : y + bh, x : x + bw] = crop
                    masks.append(
                        SegMask(
                            label=label,
                            mask=full_mask,
                            confidence_mask=conf,
                            bbox=(x, y, bw, bh),
                            extras={"source": "segformer-hf"},
                        )
                    )
            return masks
        except Exception as exc:
            if not self._warned:
                warnings.warn(f"SegFormer HF path failed, falling back: {exc}")
                self._warned = True
            return self._fallback_detect(image)

    def _fallback_detect(self, image: np.ndarray) -> list[SegMask]:
        h, w = image.shape
        masks: list[SegMask] = []
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        lap = cv2.Laplacian(blur, cv2.CV_32F)
        edge_energy = cv2.GaussianBlur(np.abs(lap).astype(np.float32), (9, 9), 0)
        edge_energy = (edge_energy / (edge_energy.max() + 1e-6) * 255).astype(np.uint8)
        bright_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        texture_mask = cv2.threshold(edge_energy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        combined = cv2.bitwise_or(bright_mask, texture_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(80, int(h * w * 0.0015))
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < min_area or bw > w * 0.7 or bh > h * 0.6:
                continue
            crop_img = image[y : y + bh, x : x + bw]
            crop_combined = combined[y : y + bh, x : x + bw]
            if crop_img.size == 0:
                continue
            brightness = float(np.mean(crop_img))
            aspect = bw / max(bh, 1)
            label = _heuristic_label(aspect, brightness, area)
            mask_area_pct = float(np.count_nonzero(crop_combined)) / float(crop_combined.size)
            conf = float(min(0.45 + mask_area_pct * 0.5 + (brightness / 510.0), 0.95))
            full_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(full_mask, [c - np.array([x, y])], 0, 1, thickness=-1, offset=(x, y))
            if full_mask.sum() == 0:
                full_mask[y : y + bh, x : x + bw] = crop_combined // 255
            if full_mask[y : y + bh, x : x + bw].sum() < max(4, int(0.1 * bw * bh)):
                ellipse_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(
                    ellipse_mask,
                    (x + bw // 2, y + bh // 2),
                    (max(2, bw // 3), max(2, bh // 3)),
                    0,
                    0,
                    360,
                    1,
                    thickness=-1,
                )
                rect_mask = np.zeros((h, w), dtype=np.uint8)
                pad_x, pad_y = max(1, bw // 8), max(1, bh // 8)
                cv2.rectangle(
                    rect_mask,
                    (x + pad_x, y + pad_y),
                    (x + bw - pad_x, y + bh - pad_y),
                    1,
                    thickness=-1,
                )
                combined_fallback = cv2.bitwise_or(ellipse_mask, rect_mask)
                full_mask = cv2.bitwise_or(full_mask, combined_fallback)
            masks.append(
                SegMask(
                    label=label,
                    mask=full_mask,
                    confidence_mask=round(conf, 4),
                    bbox=(x, y, bw, bh),
                    extras={"source": "segformer-opencv-fallback", "mask_area_pct": round(mask_area_pct, 4)},
                )
            )
        shadow_thresh = cv2.threshold(blur, max(20, blur.mean() * 0.55), 255, cv2.THRESH_BINARY_INV)[1]
        shadow_clean = cv2.morphologyEx(shadow_thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        for _ in range(2):
            shadow_clean = cv2.dilate(shadow_clean, np.ones((3, 3), np.uint8), iterations=1)
        return masks


def _heuristic_label(aspect: float, brightness: float, area: int) -> str:
    if aspect > 2.4:
        return "plastic_net"
    if 0.75 <= aspect <= 1.25 and brightness > 120:
        return "metal_barrel"
    if 0.8 <= aspect <= 1.4:
        return "tire"
    if aspect < 0.7:
        return "wood_wreck"
    if brightness > 130:
        return "unclassified"
    return "anomaly"
