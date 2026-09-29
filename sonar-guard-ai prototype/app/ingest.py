from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SonarFrame:
    image: np.ndarray
    source_path: Path | None
    along_m: float
    across_m: float
    gps_records: list[dict] | None = None
    gps_origin_lat: float | None = None
    gps_origin_lon: float | None = None
    xtf_ping_count: int | None = None


def load_sonar_image(path: str | Path) -> SonarFrame:
    """Stage 1: ingest a waterfall image (TIF/PNG). XTF parsing comes next."""
    path = Path(path)
    if path.suffix.lower() in (".xtf",):
        return load_xtf(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read sonar image: {path}")
    h, w = image.shape
    return SonarFrame(image=image, source_path=path, along_m=float(h), across_m=float(w))


def decode_upload(data: bytes, filename: str = "upload.png") -> SonarFrame:
    ext = Path(filename).suffix.lower()
    if ext in (".xtf",):
        return decode_xtf_upload(data, filename)
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unsupported image payload")
    h, w = image.shape
    return SonarFrame(image=image, source_path=Path(filename), along_m=float(h), across_m=float(w))


def load_xtf(path: str | Path) -> SonarFrame:
    from app.xtf_parser import parse_xtf

    path = Path(path)
    result = parse_xtf(path)
    h, w = result.image.shape
    return SonarFrame(
        image=result.image,
        source_path=path,
        along_m=float(h),
        across_m=float(w),
        gps_records=result.gps_records,
        gps_origin_lat=result.origin_lat,
        gps_origin_lon=result.origin_lon,
        xtf_ping_count=result.ping_count,
    )


def decode_xtf_upload(data: bytes, filename: str = "upload.xtf") -> SonarFrame:
    from app.xtf_parser import parse_xtf

    result = parse_xtf(data)
    h, w = result.image.shape
    return SonarFrame(
        image=result.image,
        source_path=Path(filename),
        along_m=float(h),
        across_m=float(w),
        gps_records=result.gps_records,
        gps_origin_lat=result.origin_lat,
        gps_origin_lon=result.origin_lon,
        xtf_ping_count=result.ping_count,
    )
