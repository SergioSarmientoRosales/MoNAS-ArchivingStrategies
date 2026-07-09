from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from monas_archiving.config import load_config
from monas_archiving.pipeline import run_pipeline


def write_config(input_csv: Path, output_dir: Path) -> Path:
    config_path = output_dir / "all_archivers_config.yaml"
    config = {
        "run_name": "run_all_archivers",
        "input_path": str(input_csv),
        "output_dir": str(output_dir.parent),
        "architecture_id_column": "architecture_id",
        "chromosome_column": "chromosome",
        "deduplication_key": "architecture_id",
        "seed": 3,
        "objectives": [
            {"column": "psnr", "direction": "maximize"},
            {"column": "params", "direction": "minimize"},
        ],
        "archivers": [
            {"name": "pq"},
            {"name": "hv"},
            {"name": "r2"},
            {"name": "crowding"},
            {"name": "grid", "bins": 4},
            {"name": "epsilon", "eps": 0.2},
            {"name": "tight1"},
            {"name": "kmeans", "iters": 30},
            {"name": "entropy", "bins": 4},
        ],
        "truncation_sizes": [3, 5],
        "indicators": ["igd_plus", "hypervolume", "r2", "epsilon", "hausdorff"],
        "hv_reference_point": [1.1, 1.1],
        "plot": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all archivers on examples/example_input.csv.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "examples" / "example_input.csv",
        help="Small input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "examples",
        help="Directory where example outputs will be written.",
    )
    args = parser.parse_args()

    run_root = args.output_dir / "run_all_archivers"
    config_path = write_config(args.input, run_root)
    config = load_config(config_path)
    run_pipeline(config)
    metrics = pd.read_csv(config.run_dir / "metrics" / "archive_metrics.csv")
    print(metrics[["archiver", "k", "indicator", "value", "direction"]].to_string(index=False))
    print(f"Outputs written to {config.run_dir}")


if __name__ == "__main__":
    main()
