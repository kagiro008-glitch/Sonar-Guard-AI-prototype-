import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import OUTPUTS_DIR, SAMPLES_DIR, SURVEY_ORIGIN_LAT, SURVEY_ORIGIN_LON
from app.pipeline import SonarGuardPipeline

app = FastAPI(title="SonarGuard AI", version="0.2.0")
pipeline = SonarGuardPipeline()
DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"

if DASHBOARD.exists():
    app.mount("/static", StaticFiles(directory=DASHBOARD), name="static")


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    index = DASHBOARD / "index.html"
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/flowchart", response_class=HTMLResponse)
def flowchart() -> HTMLResponse:
    page = DASHBOARD / "autoencoder_flowchart.html"
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "sonar-guard-ai", "version": "0.2.0"}


@app.get("/api/map-config")
def map_config() -> dict:
    return {
        "lat": SURVEY_ORIGIN_LAT,
        "lon": SURVEY_ORIGIN_LON,
        "note": "Demo GPS is a Bay of Bengal survey line off Chennai until real XTF navigation is parsed. "
        "Upload .xtf files for embedded GPS.",
    }


@app.get("/api/samples")
def list_samples() -> dict:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in SAMPLES_DIR.glob("*.png") if not p.name.endswith("_mask.png"))
    xtf_files = sorted(p.name for p in SAMPLES_DIR.glob("*.xtf"))
    return {"samples": files, "xtf_samples": xtf_files}


@app.post("/api/analyze/sample/{name}")
def analyze_sample(name: str) -> JSONResponse:
    path = SAMPLES_DIR / name
    if not path.exists():
        alt = SAMPLES_DIR / (name + ".png")
        if alt.exists():
            path = alt
        else:
            raise HTTPException(status_code=404, detail="Sample not found")
    result = pipeline.run_path(path)
    return JSONResponse({"stats": result.stats, "detections": result.detections, "geojson": result.geojson})


@app.post("/api/analyze/upload")
async def analyze_upload(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    result = pipeline.run_bytes(data, file.filename or "upload.png")
    return JSONResponse({"stats": result.stats, "detections": result.detections, "geojson": result.geojson})


@app.get("/api/preview")
def preview() -> Response:
    path = OUTPUTS_DIR / "latest_preview.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run an analysis first")
    return FileResponse(path, media_type="image/png")


@app.get("/api/geojson")
def geojson() -> JSONResponse:
    path = OUTPUTS_DIR / "latest.geojson"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run an analysis first")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/benchmark/last")
def benchmark_last() -> JSONResponse:
    path = OUTPUTS_DIR / "benchmark_last.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No benchmark report yet — POST /api/benchmark/run")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.post("/api/benchmark/run")
def benchmark_run() -> JSONResponse:
    from app.benchmark import run_benchmark

    report = run_benchmark()
    return JSONResponse(report)


@app.post("/api/model/export-trt")
def export_trt() -> JSONResponse:
    from app.export import export_all

    result = export_all()
    return JSONResponse(result)
