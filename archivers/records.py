from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Record:
    model: str
    seed: int
    net: str
    psnr: float
    params: float
    generation: int          # NEW
    source_file: str
    meta: Any = None

@dataclass(frozen=True)
class NormRecord:
    base: Record
    psnr_n: float
    params_n: float
