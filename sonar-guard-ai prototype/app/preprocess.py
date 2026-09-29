from __future__ import annotations

import cv2
import numpy as np


def apply_tvg(image: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Time-varied gain: brighten farther ranges (columns) to offset attenuation."""
    h, w = image.shape
    gain = 1.0 + strength * np.linspace(0.0, 1.0, w, dtype=np.float32)
    boosted = image.astype(np.float32) * gain[None, :]
    return np.clip(boosted, 0, 255).astype(np.uint8)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    return clahe.apply(image)


def despeckle(image: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(image, 5)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Stage 2: TVG + CLAHE + speckle filter."""
    return despeckle(apply_clahe(apply_tvg(image)))
