from __future__ import annotations
from typing import Iterable, List
from records import Record, NormRecord

class GlobalMinMaxNormalizer:
    def __init__(self, psnr_min: float, psnr_max: float, params_min: float, params_max: float):
        self.psnr_min = psnr_min
        self.psnr_max = psnr_max
        self.params_min = params_min
        self.params_max = params_max

    @staticmethod
    def fit(records: Iterable[Record]) -> "GlobalMinMaxNormalizer":
        rs = list(records)
        if not rs:
            raise ValueError("No records provided for normalization.")
        return GlobalMinMaxNormalizer(
            psnr_min=min(r.psnr for r in rs),
            psnr_max=max(r.psnr for r in rs),
            params_min=min(r.params for r in rs),
            params_max=max(r.params for r in rs),
        )

    @staticmethod
    def _mm(x: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.0
        return (x - lo) / (hi - lo)

    def transform(self, records: Iterable[Record]) -> List[NormRecord]:
        out: List[NormRecord] = []
        for r in records:
            out.append(NormRecord(
                base=r,
                psnr_n=self._mm(r.psnr, self.psnr_min, self.psnr_max),
                params_n=self._mm(r.params, self.params_min, self.params_max),
            ))
        return out
