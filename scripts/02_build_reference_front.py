from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from monas_archiving.config import load_config
from monas_archiving.pipeline import build_reference_front


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the empirical reference Pareto front.")
    parser.add_argument("--config", required=True, help="Path to a YAML pipeline config.")
    args = parser.parse_args()
    config = load_config(args.config)
    front = build_reference_front(config)
    print(f"Wrote {len(front)} reference-front solutions to {config.run_dir / 'reference_front.csv'}")


if __name__ == "__main__":
    main()
