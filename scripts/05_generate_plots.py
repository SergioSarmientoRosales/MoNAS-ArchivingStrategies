from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from monas_archiving.config import load_config
from monas_archiving.pipeline import generate_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lightweight pipeline plots.")
    parser.add_argument("--config", required=True, help="Path to a YAML pipeline config.")
    args = parser.parse_args()
    config = load_config(args.config)
    generate_plots(config)
    print(f"Plotting stage complete. See {config.run_dir / 'figures'}")


if __name__ == "__main__":
    main()
