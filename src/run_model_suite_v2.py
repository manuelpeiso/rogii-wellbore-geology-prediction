from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from train_model_v2 import MODEL_TYPES_V2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODEL_TYPES_V2, choices=MODEL_TYPES_V2)
    parser.add_argument("--max-train-rows", type=int, default=3_000_000)
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/model_results_v2.csv"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    for model_type in args.models:
        print(f"\n=== {model_type} ===", flush=True)
        cmd = [
            sys.executable,
            "src/train_model_v2.py",
            "--model-type",
            model_type,
            "--max-train-rows",
            str(args.max_train_rows),
            "--metrics-output",
            str(args.metrics_output),
            "--random-state",
            str(args.random_state),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
