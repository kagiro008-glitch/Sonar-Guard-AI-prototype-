[README.md](https://github.com/user-attachments/files/32795422/README.md)
# SonarGuard AI

Production-ready prototype for **SIH 2026 · SIH26057** — AI-powered marine debris and anomaly detection from side-scan sonar. Team: **Deep Horizon**.

Implements the full five-stage flow from the pitch deck, plus:
- Native `.xtf` (XTF Triton Format) ingestion with embedded GPS waypoints
- **SegFormer** semantic segmentation branch (OpenCV fallback before fine-tuned weights arrive)
- **TensorRT / Jetson** export pipeline (ONNX + trtexec build hooks)
- **Advanced synthetic data generator** with segmentation GT masks and GPS manifests
- **Benchmark suite** with per-stage latencies, severity ranking, and <10 cm geotag-precision budget report
- Upgraded operator dashboard: severity badges, stage-latency strip, precision badge, ping-indexed GPS popups

## What this prototype does

1. **Ingest** grayscale sonar waterfall images (PNG / TIF) or native `.xtf` logs with embedded GPS records.
2. **Preprocess** with TVG, CLAHE, and median despeckle.
3. **Detect** known-looking debris (YOLOv11 if `weights/sonar_yolo11.pt` exists, else OpenCV blob fallback).
4. **Segment** with SegFormer family (Hugging Face `transformers` path, or OpenCV texture-classifier fallback).
5. **Flag anomalies** with a PyTorch autoencoder (reconstruction-error outliers).
6. **Validate** acoustic shadows, then fuse branches and emit severity-ranked **GeoJSON** to a live dashboard map.
7. **Benchmark** and export ONNX/TensorRT engines for NVIDIA Jetson edge deployment.

## Run locally

```powershell
cd "$env:USERPROFILE\sonar-guard-ai"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\generate_synthetic_sonar.py          # classic samples (or use --advanced --seed 42 --count 3 --with-masks)
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 and click **Analyze sample**.

## CLI

Analyze one file:
```powershell
python -m app.cli data\samples\sss_mixed_03.png     # PNG/TIF
python -m app.cli --xtf data\samples\log_001.xtf     # XTF log with embedded GPS
```

Benchmark suite:
```powershell
python -m app.cli --benchmark --benchmark-runs 5 --benchmark-warmup 1
# or:  python -m app.benchmark
```

TensorRT / ONNX export:
```powershell
python -m app.cli --export-trt
# Writes weights/export/{autoencoder,segformer_fallback}.onnx, manifest.json, and .engine if trtexec is on PATH.
```

Advanced synthetic data:
```powershell
python scripts\generate_synthetic_sonar.py --advanced --seed 42 --count 3 --with-masks
# Produces sss_adv_*.png + *_mask.png + samples_manifest.json/.yaml
```

## REST API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard |
| `/api/health` | GET | Service version + liveness |
| `/api/map-config` | GET | Survey origin and notes |
| `/api/samples` | GET | List PNG + XTF samples available |
| `/api/analyze/sample/{name}` | POST | Run pipeline on sample |
| `/api/analyze/upload` | POST | Run pipeline on uploaded file (PNG/TIF/XTF) |
| `/api/preview` | GET | Latest annotated preview PNG |
| `/api/geojson` | GET | Latest GeoJSON FeatureCollection |
| `/api/benchmark/last` | GET | Last benchmark JSON report |
| `/api/benchmark/run` | POST | Run benchmark now and return report |
| `/api/model/export-trt` | POST | Export ONNX/TensorRT models, return manifest |

## Debris classes, severity & precision

Classes in `DEBRIS_CLASSES`: `plastic_net`, `metal_barrel`, `tire`, `wood_wreck`, `unclassified`. Anomaly autoencoder hits are labelled `anomaly`.

Severity (`critical | high | medium | low`) is computed from class weight, shadow consistency, detection score, and estimated m² size — see [compute_severity](app/pipeline.py#L125-L148).

Geotag precision budget defaults to `sqrt(2) × max(METERS_PER_PIXEL_ALONG, METERS_PER_PIXEL_ACROSS)`. With the out-of-box
`METERS_PER_PIXEL_ACROSS=1.0 m` this is ≈ 141 cm — tighten these values in `app/config.py` once you calibrate against real sonar pixel resolution
to demonstrate the **< 10 cm target** claimed on slide 2 of the SIH pitch deck.

## Next training steps (what wins SIH)

1. Download public SSS debris / wreck data (Valdenegro 2025, Sethuraman 2024 shipwreck SSS, Duras et al. 2024).
2. Label a small YOLO set: `plastic_net`, `metal_barrel`, `tire`, `wood_wreck`.
3. Train `yolo11n` and save to `weights/sonar_yolo11.pt`.
4. Fine-tune `nvidia/segformer-b0` on your SSS mask GT from `--advanced --with-masks` plus public labels.
5. Train the autoencoder on **empty seabed** patches only, then raise the anomaly threshold.
6. Replace placeholder pixel→metre values in `config.py` with calibrated sonar resolution to hit the <10 cm geotag target.
7. Run `--export-trt` with `trtexec` on an NVIDIA Jetson Orin and benchmark p95 end-to-end latency.

Do not quote pitch-deck percentages as measured results until this pipeline produces them.
