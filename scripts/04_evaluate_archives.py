from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from monas_archiving.config import load_config
from monas_archiving.pipeline import evaluate_archives


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate archives against the reference front.")
    parser.add_argument("--config", required=True, help="Path to a YAML pipeline config.")
    args = parser.parse_args()
    config = load_config(args.config)
    metrics = evaluate_archives(config)
    print(f"Wrote {len(metrics)} metric rows to {config.run_dir / 'metrics' / 'archive_metrics.csv'}")


if __name__ == "__main__":
    main()
