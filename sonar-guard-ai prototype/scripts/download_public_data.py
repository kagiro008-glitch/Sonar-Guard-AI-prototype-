from __future__ import annotations

import argparse
import shutil
import urllib.request
import urllib.parse
import socket
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "samples"
CACHE_DIR = ROOT / "data" / "cache"

NOAA_HUDSON_URL = (
    "ftp://ftp.coast.noaa.gov/pub/benthic/"
    "Side_Scan_Sonar_Backscatter_Data/NY_HudsonRiver_sss-xtf.zip"
)


DATA_SOURCES: dict[str, dict] = {
    "noaa-hudson-small": {
        "description": (
            "NOAA 2009 Hudson River sidescan XTF bulk subset. "
            "Warning: full zip can be multi-GB; we attempt a small HTTP mirror first."
        ),
        "urls": [
            "https://github.com/robintibor/sss-data/raw/main/samples/NOAA_Hudson_Example.XTF",
            NOAA_HUDSON_URL,
        ],
        "fallback_generator": "survey_chennai_small",
    },
    "chennai-survey-sim": {
        "description": (
            "Generated realistic Bay-of-Bengal (Chennai coast) survey XTF "
            "with real sonar resolution and GPS track."
        ),
        "generator": "survey_chennai_realistic",
    },
}


def download(url: str, dest: Path, timeout: int = 30) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SonarGuardAI/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            with dest.open("wb") as f:
                if total is None:
                    shutil.copyfileobj(resp, f)
                else:
                    copied = 0
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        copied += len(chunk)
                        pct = 100.0 * copied / total
                        print(f"  {copied // 1024:>7d} KB / {total // 1024:>7d} KB  "
                              f"({pct:5.1f}%)", end="\r")
        print(f"\n  => OK, size={dest.stat().st_size // 1024} KB")
        return True
    except (urllib.error.URLError, socket.timeout, OSError, ValueError) as exc:
        print(f"  => FAILED: {exc.__class__.__name__}: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def extract_zip_first_xtf(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".xtf")]
            for n in names[:4]:
                target = out_dir / Path(n).name
                with zf.open(n) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(target)
                print(f"  extracted XTF -> {target.name} ({target.stat().st_size // 1024} KB)")
    except zipfile.BadZipFile as exc:
        print(f"  zip is invalid/empty, skipping extract: {exc}")
    return extracted


def build_chennai_survey_xtf(out_path: Path, realistic: bool = True) -> Path:
    import sys as _sys

    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    from app.xtf_parser import write_realistic_survey_xtf

    return write_realistic_survey_xtf(out_path, realistic=realistic)


def available() -> list[str]:
    return sorted(DATA_SOURCES.keys())


def run(dataset: str, force: bool = False) -> list[Path]:
    src = DATA_SOURCES.get(dataset)
    if src is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {available()}")

    print(f"[{dataset}] {src['description']}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []

    generator = src.get("generator")
    if generator:
        out = SAMPLES_DIR / f"{dataset}.xtf"
        if not out.exists() or force:
            print(f"  generating synthetic survey -> {out.name}")
            if generator == "survey_chennai_realistic":
                build_chennai_survey_xtf(out, realistic=True)
        else:
            print(f"  using cached {out.name}")
        results.append(out)
        return results

    urls = src.get("urls", [])
    for url in urls:
        parsed = urllib.parse.urlparse(url)
        cache_name = dataset + Path(parsed.path).suffix or ".bin"
        cached = CACHE_DIR / cache_name
        if not cached.exists() or force:
            print(f"  downloading {url}")
            ok = download(url, cached)
            if not ok:
                continue
        else:
            print(f"  cached: {cached.name}")

        if cached.suffix.lower() == ".zip":
            results.extend(extract_zip_first_xtf(cached, SAMPLES_DIR))
        elif cached.suffix.lower() == ".xtf":
            dest = SAMPLES_DIR / cached.name
            if not dest.exists() or force:
                shutil.copy2(cached, dest)
            results.append(dest)
        else:
            results.append(cached)

        if results:
            break

    if not results:
        fallback = src.get("fallback_generator")
        if fallback:
            print(f"  remote download failed; falling back to generator '{fallback}'")
            out = SAMPLES_DIR / f"{dataset}_fallback.xtf"
            build_chennai_survey_xtf(out, realistic=True)
            results.append(out)

    return results


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Download public XTF sonar datasets for SonarGuard AI")
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="List available dataset keys")

    p_get = sub.add_parser("get", help="Fetch a dataset (download or generate)")
    p_get.add_argument("dataset", help="Dataset key or 'all'")
    p_get.add_argument("--force", action="store_true", help="Redownload / regenerate even if cached")

    args = parser.parse_args()
    if args.cmd == "list":
        print("Available datasets:")
        for k in available():
            print(f"  - {k:25s}  {DATA_SOURCES[k]['description']}")
        return
    if args.cmd in (None, "get"):
        key = getattr(args, "dataset", None)
        if not key:
            parser.print_help()
            return
        keys = available() if key == "all" else [key]
        all_results: list[Path] = []
        for k in keys:
            all_results.extend(run(k, force=getattr(args, "force", False)))
        print(f"\nResulting XTF sample files ({len(all_results)}):")
        for p in all_results:
            print(f"  - {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    _cli()
