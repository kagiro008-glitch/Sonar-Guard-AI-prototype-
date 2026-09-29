from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = DATA_DIR / "outputs"
WEIGHTS_DIR = ROOT / "weights"

SURVEY_ORIGIN_LAT = 13.0480
SURVEY_ORIGIN_LON = 80.4120
METERS_PER_PIXEL_ALONG = 2.0
METERS_PER_PIXEL_ACROSS = 1.0

MPP_OVERRIDES: dict[str, dict[str, float]] = {
    "sss_adv_nets_01.png": {"along": 0.04, "across": 0.05},
    "sss_adv_barrels_02.png": {"along": 0.04, "across": 0.05},
    "sss_adv_mixed_03.png": {"along": 0.04, "across": 0.05},
    "chennai-survey-sim.xtf": {"along": 0.05, "across": 0.04},
    "chennai_hr_400khz_seedA.xtf": {"along": 0.05, "across": 0.04},
    "chennai_hr_400khz_seedB.xtf": {"along": 0.05, "across": 0.04},
    "chennai_hr_400khz_seedC.xtf": {"along": 0.05, "across": 0.04},
    "demo_log_001.xtf": {"along": 0.20, "across": 0.10},
}


def get_mpp(sample_name: str | None = None) -> tuple[float, float]:
    if sample_name and sample_name in MPP_OVERRIDES:
        ov = MPP_OVERRIDES[sample_name]
        return float(ov["along"]), float(ov["across"])
    return METERS_PER_PIXEL_ALONG, METERS_PER_PIXEL_ACROSS


DEBRIS_CLASSES = [
    "plastic_net",
    "metal_barrel",
    "tire",
    "wood_wreck",
    "unclassified",
]

ANOMALY_THRESHOLD = 0.35
YOLO_CONFIDENCE = 0.25
SHADOW_BOOST = 0.15
SHADOW_PENALTY = 0.20
