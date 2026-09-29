from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"

DEBRIS_CLASS_IDX = {
    "plastic_net": 1,
    "metal_barrel": 2,
    "tire": 3,
    "wood_wreck": 4,
    "unclassified": 5,
}
CLASS_ORDER = ["plastic_net", "metal_barrel", "tire", "wood_wreck"]


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _add_multipath_ghost(canvas: np.ndarray, obj_y: int, obj_x: int, bw: int, bh: int, rng: np.random.Generator) -> None:
    h, w = canvas.shape
    dy = rng.integers(20, 60)
    dx = rng.integers(-40, 40)
    gy, gx = obj_y + dy, obj_x + dx
    if 0 <= gy < h and 0 <= gx < w:
        gh = min(bh, h - gy)
        gw = min(bw, w - gx)
        if gh > 4 and gw > 4:
            overlay = canvas[gy : gy + gh, gx : gx + gw].astype(np.float32)
            overlay += rng.uniform(20, 45, (gh, gw)).astype(np.float32)
            canvas[gy : gy + gh, gx : gx + gw] = np.clip(overlay, 0, 255).astype(np.uint8)


def make_waterfall(seed: int, kind: str) -> tuple[np.ndarray, list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    h, w = 640, 960
    seabed = rng.normal(48, 14, (h, w)).astype(np.float32)
    for y in range(h):
        seabed[y] += 6 * np.sin(y / 28.0)
    seabed += np.linspace(0, 18, w)[None, :]
    objects = {
        "nets": [(180, 220, 160, 28, 210, "plastic_net")],
        "barrels": [(320, 410, 42, 48, 230, "metal_barrel"), (470, 620, 38, 44, 220, "tire")],
        "mixed": [
            (140, 160, 150, 24, 205, "plastic_net"),
            (360, 500, 46, 50, 235, "metal_barrel"),
            (500, 300, 40, 40, 200, "wood_wreck"),
        ],
    }[kind]
    canvas = np.clip(seabed, 0, 255)
    boxes = []
    gps_waypoints = []
    base_lat = 13.0480
    base_lon = 80.4120
    for ping in range(0, h, 64):
        gps_waypoints.append(
            {
                "lat": round(base_lat + (ping / h) * 0.0012, 7),
                "lon": round(base_lon + (ping / h) * 0.0006, 7),
                "ping": ping,
            }
        )
    for y, x, bw, bh, intensity, cls_name in objects:
        canvas[y : y + bh, x : x + bw] = np.maximum(
            canvas[y : y + bh, x : x + bw], np.float32(intensity)
        )
        shadow_w = int(bw * 0.9)
        sx = x + bw
        shadow = canvas[y : y + bh, sx : sx + shadow_w] * 0.35
        canvas[y : y + bh, sx : sx + shadow_w] = shadow
        boxes.append({"y": y, "x": x, "w": bw, "h": bh, "class": cls_name})
    speckle = rng.normal(0, 8, canvas.shape)
    image = np.clip(canvas + speckle, 0, 255).astype(np.uint8)
    return image, boxes, gps_waypoints


def make_waterfall_advanced(
    seed: int,
    kind: str,
    speckle_std: float = 11.0,
    add_ghost: bool = True,
    rotate_objects: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    h, w = 640, 960
    seabed = rng.normal(52, 16, (h, w)).astype(np.float32)
    for y in range(h):
        seabed[y] += 5.5 * np.sin(y / 24.0) + 2.2 * np.sin(y / 9.0)
    gain = np.power(np.linspace(0.55, 1.2, w, dtype=np.float32), 1.2)
    seabed *= gain[None, :]
    nadir = np.linspace(0.65, 1.0, w // 2, dtype=np.float32)
    nadir = np.concatenate([nadir[::-1], nadir])[:w]
    seabed *= nadir[None, :]
    objects = {
        "nets": [(180, 220, 160, 28, 215, "plastic_net"), (450, 520, 130, 22, 205, "plastic_net")],
        "barrels": [(320, 410, 42, 48, 232, "metal_barrel"), (470, 620, 38, 44, 222, "tire"), (200, 700, 40, 46, 218, "metal_barrel")],
        "mixed": [
            (120, 160, 150, 24, 208, "plastic_net"),
            (360, 500, 46, 50, 238, "metal_barrel"),
            (500, 300, 40, 40, 202, "wood_wreck"),
            (260, 720, 34, 38, 214, "tire"),
        ],
    }[kind]
    canvas = np.clip(seabed, 0, 255)
    mask = np.zeros((h, w), dtype=np.uint8)
    boxes = []
    gps_waypoints = []
    base_lat = 13.0480
    base_lon = 80.4120
    for ping in range(0, h, 32):
        gps_waypoints.append(
            {
                "lat": round(base_lat + (ping / h) * 0.0012 + rng.uniform(-1e-6, 1e-6), 7),
                "lon": round(base_lon + (ping / h) * 0.0006 + rng.uniform(-1e-6, 1e-6), 7),
                "ping": ping,
            }
        )
    for obj_i, (y, x, bw, bh, intensity, cls_name) in enumerate(objects):
        obj_canvas = np.zeros((bh + 40, bw + 40), dtype=np.float32)
        oy, ox = 20, 20
        obj_canvas[oy : oy + bh, ox : ox + bw] = intensity
        if rotate_objects:
            angle = rng.uniform(-18.0, 18.0)
            obj_canvas = _rotate_image(obj_canvas.astype(np.uint8), angle).astype(np.float32)
        new_ys, new_xs = np.where(obj_canvas > max(20, intensity - 30))
        if new_ys.size == 0:
            ny1, ny2, nx1, nx2 = oy, oy + bh, ox, ox + bw
        else:
            ny1, ny2 = int(new_ys.min()), int(new_ys.max()) + 1
            nx1, nx2 = int(new_xs.min()), int(new_xs.max()) + 1
        nbw, nbh = nx2 - nx1, ny2 - ny1
        ty = max(0, min(h - nbh, y + oy - ny1))
        tx = max(0, min(w - nbw, x + ox - nx1))
        shape = (min(nbh, h - ty), min(nbw, w - tx))
        dest_slice_obj = obj_canvas[ny1 : ny1 + shape[0], nx1 : nx1 + shape[1]]
        dest_slice_canvas = canvas[ty : ty + shape[0], tx : tx + shape[1]]
        canvas[ty : ty + shape[0], tx : tx + shape[1]] = np.maximum(dest_slice_canvas, dest_slice_obj).astype(np.uint8)
        cls_idx = DEBRIS_CLASS_IDX.get(cls_name, 5)
        mask_dest = mask[ty : ty + shape[0], tx : tx + shape[1]]
        mask_dest[dest_slice_obj > max(20, intensity - 30)] = cls_idx
        shadow_len = int(bw * rng.uniform(0.6, 1.1))
        sx = tx + shape[1]
        sh, sw = shape[0], min(shadow_len, w - sx)
        if sw > 4 and sh > 4:
            shadow = canvas[ty : ty + sh, sx : sx + sw].astype(np.float32) * rng.uniform(0.28, 0.48)
            canvas[ty : ty + sh, sx : sx + sw] = np.clip(shadow, 0, 255).astype(np.uint8)
        if add_ghost and rng.random() < 0.7:
            _add_multipath_ghost(canvas, ty, tx, shape[1], shape[0], rng)
        boxes.append({"y": ty, "x": tx, "w": shape[1], "h": shape[0], "class": cls_name})
    speckle = rng.normal(0, speckle_std, canvas.shape)
    image = np.clip(canvas + speckle, 0, 255).astype(np.uint8)
    return image, mask, boxes, gps_waypoints


def _write_manifest(entries: dict, out_dir: Path) -> Path:
    manifest = out_dir / "samples_manifest.json"
    try:
        import yaml  # type: ignore

        yaml_path = out_dir / "samples_manifest.yaml"
        yaml_path.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")
    except Exception:
        pass
    manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic side-scan sonar samples")
    parser.add_argument("--advanced", action="store_true", help="Enable StyleGAN-style augmentations and masks")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--with-masks", action="store_true", help="Emit segmentation ground-truth masks (implied by --advanced)")
    args = parser.parse_args()

    SAMPLES.mkdir(parents=True, exist_ok=True)
    kinds = ["nets", "barrels", "mixed"]
    manifest_entries: dict = {}
    if args.advanced or args.with_masks:
        with_masks = True
    else:
        with_masks = False
    count = max(1, args.count)
    for i in range(count):
        kind = kinds[i % len(kinds)]
        if args.advanced:
            image, mask, boxes, gps = make_waterfall_advanced(
                seed=args.seed + i * 97,
                kind=kind,
                speckle_std=9.0 + (i % 3) * 1.5,
                add_ghost=True,
                rotate_objects=True,
            )
            name = f"sss_adv_{kind}_{i+1:02d}.png"
            cv2.imwrite(str(SAMPLES / name), image)
            adv_mpp_along = 0.04
            adv_mpp_across = 0.05
            if with_masks:
                mask_name = f"sss_adv_{kind}_{i+1:02d}_mask.png"
                cv2.imwrite(str(SAMPLES / mask_name), mask)
                manifest_entries[name] = {
                    "mask": mask_name,
                    "boxes": boxes,
                    "simulated_gps_waypoints": gps,
                    "mpp_along": adv_mpp_along,
                    "mpp_across": adv_mpp_across,
                    "params": {
                        "mode": "advanced",
                        "seed": args.seed + i * 97,
                        "kind": kind,
                        "mpp_along": adv_mpp_along,
                        "mpp_across": adv_mpp_across,
                    },
                }
            else:
                manifest_entries[name] = {
                    "boxes": boxes,
                    "simulated_gps_waypoints": gps,
                    "mpp_along": adv_mpp_along,
                    "mpp_across": adv_mpp_across,
                    "params": {
                        "mode": "advanced",
                        "seed": args.seed + i * 97,
                        "kind": kind,
                        "mpp_along": adv_mpp_along,
                        "mpp_across": adv_mpp_across,
                    },
                }
        else:
            image, boxes, gps = make_waterfall(seed=(i + 1) * 17, kind=kind)
            name = f"sss_{kind}_{i+1:02d}.png"
            cv2.imwrite(str(SAMPLES / name), image)
            manifest_entries[name] = {
                "boxes": boxes,
                "simulated_gps_waypoints": gps,
                "mpp_along": 2.0,
                "mpp_across": 1.0,
                "params": {"mode": "classic", "seed": (i + 1) * 17, "kind": kind, "mpp_along": 2.0, "mpp_across": 1.0},
            }
    _write_manifest(manifest_entries, SAMPLES)
    print(f"Wrote synthetic SSS samples to {SAMPLES} ({count} per mode)")


if __name__ == "__main__":
    main()
