from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


class PatchAutoencoder(nn.Module):
    def __init__(self, latent: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 32 * 16 * 16),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (32, 16, 16)),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class AnomalyHit:
    x: int
    y: int
    w: int
    h: int
    score: float
    label: str = "anomaly"
    extras: dict = field(default_factory=dict)


class AnomalyBranch:
    """Unsupervised reconstruction-error detector for unseen/buried objects."""

    def __init__(self, patch: int = 64, stride: int = 32, threshold: float = 0.35) -> None:
        self.patch = patch
        self.stride = stride
        self.threshold = threshold
        self.device = torch.device("cpu")
        self.model = PatchAutoencoder().to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> list[AnomalyHit]:
        h, w = image.shape
        patches: list[tuple[int, int, float]] = []
        norm = image.astype(np.float32) / 255.0
        for y in range(0, max(h - self.patch, 1), self.stride):
            for x in range(0, max(w - self.patch, 1), self.stride):
                crop = norm[y : y + self.patch, x : x + self.patch]
                if crop.shape != (self.patch, self.patch):
                    continue
                tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0)
                recon = self.model(tensor)
                error = float(torch.mean((tensor - recon) ** 2).item())
                patches.append((x, y, error))
        if not patches:
            return []
        errors = np.array([p[2] for p in patches], dtype=np.float32)
        cutoff = max(self.threshold, float(np.median(errors) + 1.8 * (np.std(errors) + 1e-6)))
        hits: list[AnomalyHit] = []
        for x, y, error in patches:
            if error < cutoff:
                continue
            hits.append(
                AnomalyHit(
                    x=x,
                    y=y,
                    w=self.patch,
                    h=self.patch,
                    score=min(error / (cutoff * 1.5), 1.0),
                    extras={"reconstruction_error": round(error, 4)},
                )
            )
        return _nms(hits)


def _nms(hits: list[AnomalyHit], iou: float = 0.3) -> list[AnomalyHit]:
    ordered = sorted(hits, key=lambda h: h.score, reverse=True)
    kept: list[AnomalyHit] = []
    for hit in ordered:
        if all(_iou(hit, other) < iou for other in kept):
            kept.append(hit)
    return kept[:12]


def _iou(a: AnomalyHit, b: AnomalyHit) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0
