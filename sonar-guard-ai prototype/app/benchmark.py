from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from app.config import (
    DATA_DIR,
    OUTPUTS_DIR,
    SAMPLES_DIR,
    get_mpp,
)
from app.pipeline import SonarGuardPipeline, pixel_to_latlon


def precision_budget_m(along: float | None = None, across: float | None = None) -> float:
    a = along if along is not None else get_mpp(None)[0]
    c = across if across is not None else get_mpp(None)[1]
    return float(max(a, c) * np.sqrt(2))


def assert_precision_budget(along: float | None = None, across: float | None = None) -> dict:
    budget = precision_budget_m(along, across)
    target = 0.10
    meets = budget <= target
    return {
        "budget_m": round(budget, 6),
        "budget_cm": round(budget * 100.0, 3),
        "target_m": target,
        "target_cm": target * 100.0,
        "meets_10cm_target": meets,
    }


def _load_manifest(sample_dir: Path) -> dict | None:
    path = sample_dir / "samples_manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_mpp(sample_name: str, manifest_entry: dict | None) -> tuple[float, float]:
    if manifest_entry:
        a = manifest_entry.get("mpp_along")
        c = manifest_entry.get("mpp_across")
        if isinstance(a, (int, float)) and isinstance(c, (int, float)):
            return float(a), float(c)
    return get_mpp(sample_name)


def _iou(a: dict, b: dict) -> float:
    ax1, ay1 = a.get("x", 0), a.get("y", 0)
    ax2, ay2 = ax1 + a.get("w", 0), ay1 + a.get("h", 0)
    bx1, by1 = b.get("x", 0), b.get("y", 0)
    bx2, by2 = bx1 + b.get("w", 0), by1 + b.get("h", 0)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = max(1, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return float(inter) / float(union)


def _great_circle_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = float(np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2.0) ** 2)
    return float(2 * R * np.arcsin(min(1.0, np.sqrt(a))))


def _interp_gps_at_ping(gps_waypoints: list[dict], ping: int, height: int) -> tuple[float, float] | None:
    if not gps_waypoints:
        return None
    recs = sorted(
        [
            (int(r.get("ping", 0)), float(r.get("lat", 0.0)), float(r.get("lon", 0.0)))
            for r in gps_waypoints
            if "lat" in r and "lon" in r
        ],
        key=lambda x: x[0],
    )
    if not recs:
        return None
    if ping <= recs[0][0]:
        return recs[0][1], recs[0][2]
    if ping >= recs[-1][0]:
        return recs[-1][1], recs[-1][2]
    for i in range(len(recs) - 1):
        p0, lat0, lon0 = recs[i]
        p1, lat1, lon1 = recs[i + 1]
        if p0 <= ping <= p1:
            if p1 == p0:
                return lat0, lon0
            t = (ping - p0) / (p1 - p0)
            return lat0 + (lat1 - lat0) * t, lon0 + (lon1 - lon0) * t
    return None


def _pseudo_ground_truth_eval(
    detections: list[dict],
    manifest_entry: dict | None,
    image_shape: tuple[int, int],
    mpp: tuple[float, float],
) -> dict:
    if not manifest_entry:
        return {"manifest_present": False}
    h, _w = image_shape
    along, across = mpp
    gps_waypoints = manifest_entry.get("simulated_gps_waypoints") or []
    boxes_gt = manifest_entry.get("boxes") or []
    result = {
        "manifest_present": True,
        "waypoints_count": len(gps_waypoints),
        "gt_boxes_count": len(boxes_gt),
        "detections_count": len(detections),
    }
    if detections and gps_waypoints:
        deltas_m: list[float] = []
        for det in detections:
            cx = float(det.get("x", 0.0)) + float(det.get("w", 0.0)) / 2.0
            cy = float(det.get("y", 0.0)) + float(det.get("h", 0.0)) / 2.0
            ping = int(np.clip(cy, 0, max(0, h - 1)))
            sim = _interp_gps_at_ping(gps_waypoints, ping, h)
            if sim is None:
                continue
            sim_lat, sim_lon = sim
            det_lat, det_lon = pixel_to_latlon(cx, cy, h, (along, across))
            deltas_m.append(_great_circle_m(sim_lat, sim_lon, det_lat, det_lon))
        if deltas_m:
            arr = np.asarray(deltas_m, dtype=np.float64)
            result["gps_delta_rmse_m"] = round(float(np.sqrt(np.mean(arr**2))), 5)
            result["gps_delta_mean_m"] = round(float(np.mean(arr)), 5)
            result["gps_delta_max_m"] = round(float(np.max(arr)), 5)
        else:
            result["gps_delta_rmse_m"] = None
            result["gps_delta_mean_m"] = None
            result["gps_delta_max_m"] = None
    if detections and boxes_gt:
        matched = 0
        matches: list[dict] = []
        for gt in boxes_gt:
            best = max((_iou(gt, d) for d in detections), default=0.0)
            passed = best >= 0.4
            if passed:
                matched += 1
            matches.append({"iou": round(best, 4), "matched_at_04": passed})
        result["box_recall_04"] = round(float(matched) / float(len(boxes_gt)), 4)
        result["matched_boxes"] = matched
        result["total_gt_boxes"] = len(boxes_gt)
        result["per_gt_box_matches"] = matches
    return result


def run_benchmark(
    sample_dir: Path | None = None,
    warmup: int = 1,
    runs: int = 5,
) -> dict:
    sample_dir = Path(sample_dir) if sample_dir else SAMPLES_DIR
    pipeline = SonarGuardPipeline()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(sample_dir)
    pngs = sorted([p for p in sample_dir.glob("*.png") if not p.name.endswith("_mask.png")])
    per_image: dict[str, dict] = {}
    all_stage_times: dict[str, list[float]] = {}
    pgt_samples = 0
    pgt_rmse_list: list[float] = []
    pgt_recall_list: list[float] = []
    precision_samples: list[dict] = []
    for path in pngs:
        name = path.name
        manifest_entry = manifest.get(name) if manifest else None
        mpp = _resolve_mpp(name, manifest_entry)
        image_times: list[float] = []
        stage_times: dict[str, list[float]] = {}
        last_stats = None
        last_detections = None
        last_shape = None
        last_geojson = None
        for run_i in range(warmup + runs):
            t_total = time.perf_counter()
            result = pipeline.run_path(path, mpp=mpp)
            elapsed_ms = (time.perf_counter() - t_total) * 1000.0
            if run_i < warmup:
                continue
            image_times.append(elapsed_ms)
            last_stats = result.stats
            last_detections = result.detections
            last_shape = result.stats.get("image_shape") if result.stats else None
            last_geojson = result.geojson
            timings_ms = result.stats.get("timings_ms", {}) or {}
            for k, v in timings_ms.items():
                stage_times.setdefault(k, []).append(float(v))
                all_stage_times.setdefault(k, []).append(float(v))
        if not image_times:
            continue
        img_entry = {
            "runs": len(image_times),
            "total_ms_p50": round(statistics.median(image_times), 3),
            "total_ms_p95": round(_percentile(image_times, 95), 3),
            "total_ms_p99": round(_percentile(image_times, 99), 3),
            "stage_ms_p95": {k: round(_percentile(v, 95), 3) for k, v in stage_times.items()},
            "stats": {
                "known_count": last_stats.get("known_count") if last_stats else None,
                "anomaly_count": last_stats.get("anomaly_count") if last_stats else None,
                "alert_count": last_stats.get("alert_count") if last_stats else None,
                "severity_counts": last_stats.get("severity_counts") if last_stats else None,
                "precision_budget_m": last_stats.get("precision_budget_m") if last_stats else None,
            },
            "detection_count": len(last_detections) if last_detections else 0,
            "mpp_along": mpp[0],
            "mpp_across": mpp[1],
        }
        shape = None
        if last_shape and isinstance(last_shape, (list, tuple)) and len(last_shape) >= 2:
            shape = (int(last_shape[0]), int(last_shape[1]))
        pgt = _pseudo_ground_truth_eval(last_detections or [], manifest_entry, shape or (0, 0), mpp)
        img_entry["pseudo_ground_truth"] = pgt
        if pgt.get("manifest_present"):
            pgt_samples += 1
            if pgt.get("gps_delta_rmse_m") is not None:
                pgt_rmse_list.append(float(pgt["gps_delta_rmse_m"]))
            if pgt.get("box_recall_04") is not None:
                pgt_recall_list.append(float(pgt["box_recall_04"]))
        per_image[name] = img_entry
        sample_precision = assert_precision_budget(mpp[0], mpp[1])
        precision_samples.append(sample_precision)
    precision = assert_precision_budget()
    if precision_samples:
        precision["per_sample_meets_10cm_ratio"] = round(
            sum(1 for p in precision_samples if p["meets_10cm_target"]) / len(precision_samples), 4
        )
        precision["samples_meeting_10cm_count"] = sum(1 for p in precision_samples if p["meets_10cm_target"])
        precision["samples_with_precision_count"] = len(precision_samples)
    pgt_agg = {
        "manifest_present": bool(manifest),
        "samples_with_manifest_coverage": pgt_samples,
        "total_samples": len(per_image),
    }
    if pgt_rmse_list:
        pgt_agg["avg_gps_delta_rmse_m"] = round(float(np.mean(np.asarray(pgt_rmse_list, dtype=np.float64))), 5)
        pgt_agg["worst_gps_delta_rmse_m"] = round(float(np.max(np.asarray(pgt_rmse_list, dtype=np.float64))), 5)
    if pgt_recall_list:
        pgt_agg["total_box_recall_04"] = round(float(np.mean(np.asarray(pgt_recall_list, dtype=np.float64))), 4)
        pgt_agg["per_sample_recall_04"] = [round(x, 4) for x in pgt_recall_list]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_dir": str(sample_dir),
        "runs_per_sample": runs,
        "warmup_runs": warmup,
        "samples_analyzed": list(per_image.keys()),
        "per_image": per_image,
        "aggregate_stage_ms_p95": {k: round(_percentile(v, 95), 3) for k, v in all_stage_times.items()},
        "precision_budget": precision,
        "pseudo_ground_truth_aggregate": pgt_agg,
    }
    report_path = OUTPUTS_DIR / "benchmark_last.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(np.percentile(np.asarray(values, dtype=np.float32), pct))


def print_summary(report: dict) -> None:
    print(f"Samples analyzed: {len(report['samples_analyzed'])}")
    for name, info in report["per_image"].items():
        pgt = info.get("pseudo_ground_truth", {}) or {}
        pgt_note = ""
        if pgt.get("manifest_present"):
            parts = []
            if pgt.get("gps_delta_rmse_m") is not None:
                parts.append(f"gps_rmse={pgt['gps_delta_rmse_m']}m")
            if pgt.get("box_recall_04") is not None:
                parts.append(f"box_recall@0.4={pgt['box_recall_04']:.2%}")
            pgt_note = " | " + " ".join(parts) if parts else " | manifest: yes"
        print(
            f"  {name}: p50={info['total_ms_p50']}ms, "
            f"p95={info['total_ms_p95']}ms, "
            f"detections={info['detection_count']}, "
            f"mpp=({info['mpp_along']}, {info['mpp_across']})"
            f"{pgt_note}"
        )
    p = report["precision_budget"]
    print(
        f"Default geotag precision budget: {p['budget_cm']:.2f} cm "
        f"(target <10 cm: {'PASS' if p['meets_10cm_target'] else 'WARN'})"
    )
    if p.get("per_sample_meets_10cm_ratio") is not None:
        print(
            f"  Samples meeting <10 cm: {p.get('samples_meeting_10cm_count', 0)}/{p.get('samples_with_precision_count', 0)} "
            f"({p['per_sample_meets_10cm_ratio']:.0%})"
        )
    print(f"Aggregate p95 stage times (ms): {report['aggregate_stage_ms_p95']}")
    pgt_a = report.get("pseudo_ground_truth_aggregate") or {}
    if pgt_a.get("manifest_present"):
        print(
            f"Pseudo-GT manifest coverage: {pgt_a.get('samples_with_manifest_coverage', 0)}/{pgt_a.get('total_samples', 0)}"
        )
        if pgt_a.get("avg_gps_delta_rmse_m") is not None:
            print(f"  Avg GPS delta RMSE: {pgt_a['avg_gps_delta_rmse_m']} m")
        if pgt_a.get("total_box_recall_04") is not None:
            print(f"  Avg box recall@0.4: {pgt_a['total_box_recall_04']:.2%}")
    print(f"Report written to: {report.get('report_path')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SonarGuard AI benchmark suite")
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    report = run_benchmark(sample_dir=args.sample_dir, warmup=args.warmup, runs=args.runs)
    print_summary(report)


if __name__ == "__main__":
    main()
