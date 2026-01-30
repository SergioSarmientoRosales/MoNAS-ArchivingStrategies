from __future__ import annotations
from typing import List
from records import NormRecord

def dominates(a: NormRecord, b: NormRecord) -> bool:
    # PSNR: maximize, Params: minimize
    c1 = a.psnr_n >= b.psnr_n
    c2 = a.params_n <= b.params_n
    strict = (a.psnr_n > b.psnr_n) or (a.params_n < b.params_n)
    return c1 and c2 and strict
