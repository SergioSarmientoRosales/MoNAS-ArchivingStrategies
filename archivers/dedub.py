from __future__ import annotations
from typing import Dict, List
from records import Record


def pick_best_for_same_net(a: Record, b: Record) -> Record:
    """
    Selects the better record between two candidates that share the same
    network encoding (net).


    This rule is deterministic and introduces no randomness.
    """
    if a.psnr > b.psnr:
        return a
    if a.psnr < b.psnr:
        return b
    return a if a.params <= b.params else b


def deduplicate_by_net(records: List[Record]) -> List[Record]:
    """
    Deduplicates a list of Record objects by their network encoding (`net`)
    in a fully deterministic way.


    IMPORTANT:
    Although the selection itself is deterministic, the insertion order
    of dictionaries depends on the input order. Therefore, this function
    explicitly sorts the final list to guarantee a fixed and reproducible
    ordering, which is critical for offline archiving methods that are
    sensitive to insertion order (e.g., epsilon-based, grid-based, or
    greedy HV archivers).

    Sorting criteria (lexicographic, fixed):
    1) params  (ascending)  -> minimize model size
    2) psnr    (descending) -> maximize reconstruction quality
    3) net     (ascending)  -> stable tie-breaker

    With this ordering, repeated runs over the same input set will always
    produce identical archives for all deterministic offline archivers.
    """
    best: Dict[str, Record] = {}

    # Deduplication step: keep the best record per network
    for r in records:
        if r.net not in best:
            best[r.net] = r
        else:
            best[r.net] = pick_best_for_same_net(best[r.net], r)

    # Explicit deterministic ordering (critical for reproducibility)
    deduplicated = list(best.values())
    deduplicated.sort(
        key=lambda r: (
            r.params,    # smaller models first
            -r.psnr,     # higher PSNR preferred
            r.net        # stable final tie-break
        )
    )

    return deduplicated
