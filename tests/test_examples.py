from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_minimal_toy_example_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "minimal_toy_example.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Outputs written" in result.stdout
