from __future__ import annotations
import os
import re
import csv
from typing import List, Optional
from records import Record

SEED_REGEX = re.compile(r"seed[_-]?(\d+)", flags=re.IGNORECASE)

def infer_seed(path: str) -> Optional[int]:
    m = SEED_REGEX.search(path)
    return int(m.group(1)) if m else None

def iter_result_files(model_dir_path: str, exts=(".csv", ".tsv")) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(model_dir_path):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(root, fn))
    return out

def read_records_from_file(file_path: str, model: str) -> List[Record]:
    """
    Expected CSV header with at least:
      Net, PSNR, Params, Generation
    Net may contain commas and MUST be quoted in CSV.
    """
    seed = infer_seed(file_path)
    if seed is None:
        return []

    records: List[Record] = []

    with open(file_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {"Net", "PSNR", "Params", "Generation"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing required columns in {file_path}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            try:
                net = row["Net"].strip()          # FULL chromosome string
                psnr = float(row["PSNR"])
                params = float(row["Params"])
                generation = int(row["Generation"])
            except Exception:
                # malformed row
                continue

            records.append(Record(
                model=model,
                seed=seed,
                net=net,
                psnr=psnr,
                params=params,
                generation=generation,
                source_file=file_path
            ))

    return records

