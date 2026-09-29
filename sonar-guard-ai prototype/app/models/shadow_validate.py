from __future__ import annotations

import numpy as np

from app.config import SHADOW_BOOST, SHADOW_PENALTY
from app.models.yolo_branch import Detection


def validate_shadow(image: np.ndarray, det: Detection) -> Detection:
    """Stage 4: elevated objects should cast a darker shadow in the range (x) direction."""
    h, w = image.shape
    x2 = min(w, det.x + det.w)
    y1, y2 = det.y, min(h, det.y + det.h)
    shadow_w = max(int(det.w * 0.8), 8)
    sx1, sx2 = x2, min(w, x2 + shadow_w)
    object_patch = image[y1:y2, det.x:x2]
    shadow_patch = image[y1:y2, sx1:sx2]
    if object_patch.size == 0 or shadow_patch.size == 0:
        det.extras["shadow"] = "insufficient"
        return det

    object_mean = float(np.mean(object_patch))
    shadow_mean = float(np.mean(shadow_patch))
    ratio = shadow_mean / max(object_mean, 1.0)
    consistent = ratio < 0.78 and object_mean > 40
    if consistent:
        det.score = min(det.score + SHADOW_BOOST, 0.99)
        det.extras["shadow"] = "consistent"
    else:
        det.score = max(det.score - SHADOW_PENALTY, 0.05)
        det.extras["shadow"] = "inconsistent"
    det.extras["shadow_ratio"] = round(ratio, 3)
    return det
