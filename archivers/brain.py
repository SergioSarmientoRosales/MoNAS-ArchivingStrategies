# brain.py
from __future__ import annotations

import os
import csv
from typing import List

from records import Record, NormRecord
from io_readers import iter_result_files, read_records_from_file
from dedub import deduplicate_by_net
from normalize import GlobalMinMaxNormalizer
from plotting import plot_cloud_and_all_fronts, plot_all_fronts_only


from archivers.strategies.pq import PQArchiver
from archivers.strategies.eps1 import Eps1Archiver
from archivers.strategies.eps2 import Eps2Archiver
from archivers.strategies.tight1 import Tight1Archiver
from archivers.strategies.tight2 import Tight2Archiver
from archivers.strategies.pq_eps import PQEpsArchiver
from archivers.strategies.grid import GridParetoArchiver
from archivers.strategies.igd import IGDIdealArchiver
from archivers.strategies.hv import HVArchiver
from archivers.strategies.r2 import R2Archiver
from archivers.strategies.kmeans import KMeansArchiver
from archivers.strategies.crowding import CrowdingArchiver
from archivers.strategies.entropy import EntropyArchiver



# -----------------------------
# Paths / configuration
# -----------------------------
ROOT = r"G:\Mi unidad\Paper2\Seeds"

IN_ARCHIVE = os.path.join(ROOT, "c_Reduced")
OUT_DIR = os.path.join(ROOT, "Full_15_c_Pareto_Archives")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_DIRS = ["ensemble", "gbrt", "gpr", "knn", "rf", "svr", "xgb", "et"]

# -----------------------------
# I/O helpers
# -----------------------------
def write_front_csv(path: str, front: List[NormRecord]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Net", "PSNR", "Params", "Generation",
            "Model", "Seed", "SourceFile",
            "PSNR_N", "Params_N"
        ])
        for nr in front:
            r = nr.base
            w.writerow([
                r.net, r.psnr, r.params, r.generation,
                r.model, r.seed, r.source_file,
                nr.psnr_n, nr.params_n
            ])

def load_all_records() -> List[Record]:
    all_records: List[Record] = []
    for model in MODEL_DIRS:
        model_path = os.path.join(IN_ARCHIVE, model)
        if not os.path.isdir(model_path):
            print(f"[WARN] Missing model directory: {model_path}")
            continue

        files = iter_result_files(model_path, exts=(".csv", ".tsv"))
        for fp in files:
            all_records.extend(read_records_from_file(fp, model=model))

    print(f"[INFO] Loaded records: {len(all_records)}")
    return all_records

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # 1) Load
    records = load_all_records()
    if not records:
        print("[ERROR] No records loaded.")
        return

    # 2) Global dedup by Net (chromosome string)
    records = deduplicate_by_net(records)
    print(f"[INFO] After global dedup by Net: {len(records)}")

    # 3) Global normalization
    normalizer = GlobalMinMaxNormalizer.fit(records)
    nrecs = normalizer.transform(records)

    # 4) Run archivers (baseline + epsilon methods)
    archivers = {
        "PQ_baseline":  PQArchiver(max_size=100,  f1=lambda r: r.params_n, f2=lambda r: -r.psnr_n),
        "PQ":  PQArchiver(max_size=15,  f1=lambda r: r.params_n, f2=lambda r: -r.psnr_n),
        #"pq_eps": PQEpsArchiver(eps=(0.02, 0.02)),
        "Tight1": Tight1Archiver(eps=(0.009, 0.009), theta=0.5, delta_hat=0.25),
         "Eps1": Eps1Archiver(eps=(0.008,0.008), theta=0.5),
        "Grid": GridParetoArchiver(bins_x=110, bins_y=110),
        "R2": R2Archiver(n_weights=850),

        #"igd": IGDIdealArchiver(max_size=15, n_ref=100),

         "Hv": HVArchiver(max_size=20),
         "K-means": KMeansArchiver(max_size=15, iters=1000, seed=1),
         "Crowding": CrowdingArchiver(max_size=15),
         "Entropy": EntropyArchiver(max_size=15, bins_x=100, bins_y=100),

    }

    fronts_by_method = {}  # method -> list[NormRecord]

    for method, arch in archivers.items():
        arch.reset()
        arch.update(nrecs)
        front = arch.front()

        fronts_by_method[method] = front

        # CSV por método (lo dejas igual)
        out_csv = os.path.join(OUT_DIR, method, "pareto_global.csv")
        write_front_csv(out_csv, front)

    # --- AHORA, fuera del loop, haces las gráficas globales ---
    plots_global_dir = os.path.join(OUT_DIR, "plots", "GLOBAL")
    os.makedirs(plots_global_dir, exist_ok=True)

    plot_cloud_and_all_fronts(
        all_points=nrecs,
        fronts_by_method=fronts_by_method,
        out_path=os.path.join(plots_global_dir, "cloud_plus_all_fronts_raw.png"),
        title="Cloud + All Archivers (Raw)",
        use_normalized=False,
    )

    plot_all_fronts_only(
        fronts_by_method=fronts_by_method,
        out_path=os.path.join(plots_global_dir, "all_fronts_only_raw.png"),
        title="All Archivers Fronts Only (Raw)",
        use_normalized=False,
    )

    print("[DONE] Outputs written under:", OUT_DIR)

if __name__ == "__main__":
    main()

