from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from monas_archiving.config import load_config
from monas_archiving.pipeline import run_archivers


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configured offline archivers.")
    parser.add_argument("--config", required=True, help="Path to a YAML pipeline config.")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = run_archivers(config)
    print(f"Wrote {len(paths)} archives under {config.run_dir / 'archives'}")


if __name__ == "__main__":
    main()
