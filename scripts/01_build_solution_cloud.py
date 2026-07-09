from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from monas_archiving.config import load_config
from monas_archiving.pipeline import build_solution_cloud


def main() -> None:
    parser = argparse.ArgumentParser(description="Load, validate, normalize, and deduplicate a solution cloud.")
    parser.add_argument("--config", required=True, help="Path to a YAML pipeline config.")
    args = parser.parse_args()
    config = load_config(args.config)
    cloud = build_solution_cloud(config)
    print(f"Wrote {len(cloud)} deduplicated solutions to {config.run_dir / 'solution_cloud.csv'}")


if __name__ == "__main__":
    main()
