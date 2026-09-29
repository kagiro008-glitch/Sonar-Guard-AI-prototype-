"""Run: python -m app.cli data/samples/sss_mixed_03.png"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.pipeline import SonarGuardPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SonarGuard AI pipeline")
    parser.add_argument("image", nargs="?", type=Path, default=None, help="Path to sonar PNG/TIF/XTF file")
    parser.add_argument("--xtf", type=Path, default=None, help="Path to XTF log (alias for positional arg)")
    parser.add_argument("--export-trt", action="store_true", help="Export models to ONNX + optional TensorRT engine")
    parser.add_argument("--benchmark", action="store_true", help="Run the benchmark suite")
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--benchmark-warmup", type=int, default=1)
    args = parser.parse_args()

    if args.export_trt:
        from app.export import export_all

        result = export_all()
        print(json.dumps(result, indent=2))
        if any(not m.get("exported") for m in result["models"]):
            sys.exit(0)
        return

    if args.benchmark:
        from app.benchmark import print_summary, run_benchmark

        report = run_benchmark(warmup=args.benchmark_warmup, runs=args.benchmark_runs)
        print_summary(report)
        return

    target = args.xtf or args.image
    if target is None:
        parser.print_help()
        sys.exit(1)
    result = SonarGuardPipeline().run_path(target)
    print(json.dumps({"stats": result.stats, "detections": result.detections}, indent=2))


if __name__ == "__main__":
    main()
