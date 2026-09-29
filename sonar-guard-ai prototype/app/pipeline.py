from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.config import (
    METERS_PER_PIXEL_ACROSS,
    METERS_PER_PIXEL_ALONG,
    OUTPUTS_DIR,
    SURVEY_ORIGIN_LAT,
    SURVEY_ORIGIN_LON,
    get_mpp,
)
from app.ingest import SonarFrame, decode_upload, load_sonar_image
from app.models.autoencoder import AnomalyBranch, AnomalyHit
from app.models.shadow_validate import validate_shadow
from app.models.yolo_branch import Detection, YoloBranch
from app.preprocess import preprocess


@dataclass
class PipelineResult:
    detections: list[dict]
    geojson: dict
    preview_png: bytes
    stats: dict = field(default_factory=dict)


class SonarGuardPipeline:
    def __init__(self) -> None:
        self.yolo = YoloBranch()
        self.anomaly = AnomalyBranch()
        try:
            from app.models.segformer_branch import SegFormerBranch
            self.segformer = SegFormerBranch()
        except Exception as e:
            self.segformer = None

    def run_path(self, path: str | Path, mpp: tuple[float, float] | None = None) -> PipelineResult:
        p = Path(path)
        frame = load_sonar_image(p)
        if mpp is None:
            mpp = get_mpp(p.name)
        return self.run_frame(frame, mpp=mpp)

    def run_bytes(self, data: bytes, filename: str, mpp: tuple[float, float] | None = None) -> PipelineResult:
        frame = decode_upload(data, filename)
        if mpp is None:
            mpp = get_mpp(filename)
        return self.run_frame(frame, mpp=mpp)

    def run_frame(self, frame: SonarFrame, mpp: tuple[float, float] | None = None) -> PipelineResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        processed = preprocess(frame.image)
        timings["preprocess"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        yolo_raw = self.yolo.detect(processed)
        known = [validate_shadow(processed, d) for d in yolo_raw]
        timings["yolo"] = (time.perf_counter() - t0) * 1000.0
        timings["shadow"] = 0.0

        t0 = time.perf_counter()
        anomalies = self.anomaly.detect(processed)
        timings["anomaly"] = (time.perf_counter() - t0) * 1000.0

        segformer_out = None
        t0 = time.perf_counter()
        if self.segformer is not None:
            try:
                segformer_out = self.segformer.detect(processed)
            except Exception:
                segformer_out = None
        timings["segformer"] = (time.perf_counter() - t0) * 1000.0

        if segformer_out:
            known = _fuse_segformer(known, segformer_out, processed.shape)

        merged = _merge(known, anomalies, segformer_out)
        for det in merged:
            det["severity"] = compute_severity(det, mpp=mpp)
        severity_counts = _severity_histogram(merged)

        preview = _draw(processed, merged)
        shape = processed.shape

        t0 = time.perf_counter()
        geojson = to_geojson(merged, shape, frame, mpp=mpp)
        timings["geojson"] = (time.perf_counter() - t0) * 1000.0

        precision_budget_m = _precision_budget_m(mpp)
        ok, encoded = cv2.imencode(".png", preview)
        if not ok:
            raise RuntimeError("Failed to encode preview")
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / "latest.geojson").write_text(json.dumps(geojson, indent=2), encoding="utf-8")
        cv2.imwrite(str(OUTPUTS_DIR / "latest_preview.png"), preview)
        return PipelineResult(
            detections=merged,
            geojson=geojson,
            preview_png=encoded.tobytes(),
            stats={
                "known_count": sum(1 for d in merged if d["source"] != "autoencoder"),
                "anomaly_count": sum(1 for d in merged if d["source"] == "autoencoder"),
                "alert_count": sum(1 for d in merged if d["score"] >= 0.45),
                "image_shape": list(shape),
                "source": str(frame.source_path) if frame.source_path else "upload",
                "timings_ms": timings,
                "precision_budget_m": precision_budget_m,
                "severity_counts": severity_counts,
            },
        )


def _precision_budget_m(mpp: tuple[float, float] | None = None) -> float:
    along, across = mpp if mpp is not None else (METERS_PER_PIXEL_ALONG, METERS_PER_PIXEL_ACROSS)
    return max(along, across) * np.sqrt(2)


def _severity_histogram(detections: list[dict]) -> dict[str, int]:
    out = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for d in detections:
        sev = compute_severity(d)
        out[sev] = out.get(sev, 0) + 1
    return out


def compute_severity(det: dict, mpp: tuple[float, float] | None = None) -> str:
    score = float(det.get("score", 0.0))
    label = str(det.get("label", "unclassified")).lower()
    shadow = str(det.get("extras", {}).get("shadow", "insufficient")).lower()
    area_px = float(det.get("w", 0)) * float(det.get("h", 0))
    along, across = mpp if mpp is not None else (METERS_PER_PIXEL_ALONG, METERS_PER_PIXEL_ACROSS)
    area_m2 = area_px * along * across
    class_weight = {
        "plastic_net": 4.0,
        "wood_wreck": 3.5,
        "metal_barrel": 3.0,
        "tire": 2.5,
        "anomaly": 2.0,
        "unclassified": 1.5,
    }.get(label, 1.0)
    shadow_weight = 1.2 if shadow == "consistent" else 0.9 if shadow == "inconsistent" else 1.0
    size_weight = min(1.0 + np.log1p(area_m2) * 0.25, 2.0)
    index = score * class_weight * shadow_weight * size_weight
    if index >= 2.8:
        return "critical"
    if index >= 1.9:
        return "high"
    if index >= 1.1:
        return "medium"
    return "low"


def _fuse_segformer(known: list[Detection], seg_out: list, shape: tuple[int, int]) -> list[Detection]:
    h, w = shape[:2]
    for det in known:
        best_iou = 0.0
        best_mask = None
        best_label = det.label
        for seg in seg_out:
            ix1, iy1 = max(det.x, seg.bbox[0]), max(det.y, seg.bbox[1])
            ix2, iy2 = min(det.x + det.w, seg.bbox[0] + seg.bbox[2]), min(det.y + det.h, seg.bbox[1] + seg.bbox[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = det.w * det.h + seg.bbox[2] * seg.bbox[3] - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_mask = seg
                best_label = seg.label
        if best_mask is not None and best_iou > 0.0:
            mask_crop = best_mask.mask[det.y : det.y + det.h, det.x : det.x + det.w]
            if mask_crop.size > 0:
                area_pct = float(np.count_nonzero(mask_crop)) / float(mask_crop.size)
            else:
                area_pct = 0.0
            conf = float(getattr(best_mask, "confidence_mask", 0.5 + best_iou * 0.5))
            det.extras["segformer"] = {
                "mask_area_pct": round(area_pct, 4),
                "confidence_mask": round(min(conf, 1.0), 4),
                "label_vote": best_label,
                "iou": round(best_iou, 4),
            }
        else:
            det_area_pct = float(min(max(0.15, (det.w * det.h) / max(1, h * w)), 0.95))
            conf = float(min(0.55 + 0.25 * det.score, 0.92))
            seg_info = getattr(det.extras, "get", None)
            existing_seg = det.extras.get("segformer") if hasattr(det.extras, "get") else None
            det.extras["segformer"] = {
                "mask_area_pct": round(det_area_pct, 4),
                "confidence_mask": round(conf, 4),
                "label_vote": best_label,
                "iou": round(best_iou, 4),
                "fallback": "bbox_derived",
            }
    return known


def _merge(known: list[Detection], anomalies: list[AnomalyHit], seg_out: list | None = None) -> list[dict]:
    items = [_det_dict(d) for d in known]
    for hit in anomalies:
        if _overlaps_known(hit, known):
            continue
        entry = {
            "x": hit.x,
            "y": hit.y,
            "w": hit.w,
            "h": hit.h,
            "score": round(hit.score, 3),
            "label": hit.label,
            "source": "autoencoder",
            "extras": dict(hit.extras),
        }
        if seg_out:
            best_iou = 0.0
            best_label = hit.label
            best_conf = 0.0
            best_area_pct = 0.0
            for seg in seg_out:
                ix1, iy1 = max(hit.x, seg.bbox[0]), max(hit.y, seg.bbox[1])
                ix2, iy2 = min(hit.x + hit.w, seg.bbox[0] + seg.bbox[2]), min(hit.y + hit.h, seg.bbox[1] + seg.bbox[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = hit.w * hit.h + seg.bbox[2] * seg.bbox[3] - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_label = seg.label
                    best_conf = float(getattr(seg, "confidence_mask", 0.5))
                    mask_crop = seg.mask[hit.y : hit.y + hit.h, hit.x : hit.x + hit.w]
                    best_area_pct = float(np.count_nonzero(mask_crop)) / float(mask_crop.size) if mask_crop.size > 0 else 0.0
            entry["extras"]["segformer"] = {
                "mask_area_pct": round(best_area_pct, 4),
                "confidence_mask": round(best_conf, 4),
                "label_vote": best_label,
                "iou": round(best_iou, 4),
            }
        items.append(entry)
    items.sort(key=lambda d: d["score"], reverse=True)
    return items


def _det_dict(d: Detection) -> dict:
    return {
        "x": d.x,
        "y": d.y,
        "w": d.w,
        "h": d.h,
        "score": round(d.score, 3),
        "label": d.label,
        "source": d.source,
        "extras": dict(d.extras),
    }


def _overlaps_known(hit: AnomalyHit, known: list[Detection]) -> bool:
    for d in known:
        ix1, iy1 = max(hit.x, d.x), max(hit.y, d.y)
        ix2, iy2 = min(hit.x + hit.w, d.x + d.w), min(hit.y + hit.h, d.y + d.h)
        if max(0, ix2 - ix1) * max(0, iy2 - iy1) > 0.25 * hit.w * hit.h:
            return True
    return False


def to_geojson(detections: list[dict], shape: tuple[int, int], frame: SonarFrame | None = None, mpp: tuple[float, float] | None = None) -> dict:
    h, _w = shape
    along, across = mpp if mpp is not None else (METERS_PER_PIXEL_ALONG, METERS_PER_PIXEL_ACROSS)
    features = []
    gps_records = getattr(frame, "gps_records", None) if frame else None
    for i, det in enumerate(detections):
        cx = det["x"] + det["w"] / 2
        cy = det["y"] + det["h"] / 2
        if gps_records:
            from app.xtf_parser import px_to_gps as xtf_px_to_gps
            lat, lon = xtf_px_to_gps(
                ping_idx=int(cy),
                across_px=int(cx),
                gps_records=gps_records,
                frame_shape=shape,
                mpp_along=along,
                mpp_across=across,
            )
            gps_ping_index = int(cy)
            gps_ts = None
            for r in gps_records:
                if r.get("ping_index") is not None and abs(r["ping_index"] - gps_ping_index) <= 1:
                    gps_ts = r.get("timestamp")
                    break
            precision_m = _precision_budget_m((along, across))
        else:
            lat, lon = pixel_to_latlon(cx, cy, h, (along, across))
            gps_ping_index = None
            gps_ts = None
            precision_m = _precision_budget_m((along, across))
        along_m = (h - cy) * along
        across_m = cx * across
        props = {
            "id": i + 1,
            "label": det["label"],
            "score": det["score"],
            "source": det["source"],
            "shadow": det.get("extras", {}).get("shadow"),
            "alert": det["score"] >= 0.45,
            "severity": det.get("severity", "low"),
            "along_m": round(along_m, 3),
            "across_m": round(across_m, 3),
            "precision_meters": round(precision_m, 5),
        }
        if frame is not None and frame.xtf_ping_count is not None:
            props["range_from_track_m"] = round(abs(cx - _w / 2) * across, 2)
        if gps_ping_index is not None:
            props["gps_ping_index"] = gps_ping_index
        if gps_ts is not None:
            props["gps_timestamp"] = gps_ts
        seg = det.get("extras", {}).get("segformer")
        if seg:
            props["segformer_mask_pct"] = seg.get("mask_area_pct")
            props["segformer_label"] = seg.get("label_vote")
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def pixel_to_latlon(x: float, y: float, height: int, mpp: tuple[float, float] | None = None) -> tuple[float, float]:
    along, across = mpp if mpp is not None else (METERS_PER_PIXEL_ALONG, METERS_PER_PIXEL_ACROSS)
    along_m = (height - y) * along
    across_m = x * across
    dlat = along_m / 111_320.0
    dlon = across_m / (111_320.0 * np.cos(np.deg2rad(SURVEY_ORIGIN_LAT)))
    return SURVEY_ORIGIN_LAT + dlat, SURVEY_ORIGIN_LON + dlon


def _draw(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    H, W = canvas.shape[:2]
    colors = {
        "plastic_net": (40, 180, 255),
        "metal_barrel": (80, 80, 255),
        "tire": (180, 120, 40),
        "wood_wreck": (40, 200, 80),
        "unclassified": (212, 188, 0),
        "anomaly": (255, 60, 180),
    }
    sev_border = {
        "critical": (0, 0, 255),
        "high": (0, 140, 255),
        "medium": (255, 200, 0),
        "low": (120, 120, 120),
    }
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.9
    FONT_THICK = 3
    for detection_id, det in enumerate(detections, start=1):
        color = colors.get(det["label"], (255, 255, 255))
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        x = int(max(0, min(W - 1, x)))
        y = int(max(0, min(H - 1, y)))
        w = int(max(4, min(W - x, w)))
        h = int(max(4, min(H - y, h)))
        sev = det.get("severity", "low")
        border = sev_border.get(sev, color)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 4)
        cv2.rectangle(canvas, (x + 2, y + 2), (x + w - 2, y + h - 2), color, 2)
        caption = f"#{detection_id} {det['label']} {det['score']:.2f} [{sev[:1].upper()}]"
        (tw, th), baseline = cv2.getTextSize(caption, FONT, FONT_SCALE, FONT_THICK)
        pad_x, pad_y = 8, 5
        banner_w = tw + 2 * pad_x + 4
        banner_h = th + 2 * pad_y
        fits_above = y - banner_h - 4 >= 0
        fits_below = y + h + banner_h + 4 <= H
        if fits_above:
            ty_top = y - banner_h - 4
        elif fits_below:
            ty_top = y + h + 4
        else:
            ty_top = max(2, min(H - banner_h - 2, y - banner_h - 4))
        ty_baseline = ty_top + th + pad_y
        desired_x = x
        max_bx1 = 2
        max_bx2 = W - 2
        min_tx = max_bx1 + 2
        max_tx = max_bx2 - banner_w + 2
        tx = int(max(min_tx, min(max_tx, desired_x)))
        bx1 = tx - 2
        bx2 = tx + banner_w
        by1 = ty_top - 2
        by2 = ty_top + banner_h + 2
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 0), thickness=-1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, thickness=2, lineType=cv2.LINE_AA)
        line_x = tx if tx >= x + 2 else x + 2
        line_x = int(max(bx1 + 3, min(bx2 - 3, line_x)))
        cv2.line(canvas, (line_x, by1), (line_x, by2), color, thickness=2)
        cv2.putText(
            canvas,
            caption,
            (tx + pad_x, ty_baseline),
            FONT,
            FONT_SCALE,
            (0, 0, 0),
            FONT_THICK + 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            caption,
            (tx + pad_x, ty_baseline),
            FONT,
            FONT_SCALE,
            color,
            FONT_THICK,
            cv2.LINE_AA,
        )
    return canvas
