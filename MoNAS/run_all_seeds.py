import subprocess
import time
from multiprocessing import Pool
start = time.time()
TOTAL_SEEDS = 30
PARALLEL_SEEDS = 10
SCRIPT = "nsga3_model_based_2obj_compsr.py"
MODEL_PATH = "models/final_model_ensemble1.pkl"
OUTDIR = "outputs"

def run_seed(seed):
    cmd = [
        "python", SCRIPT,
        "--seed", str(seed),
        "--model-path", MODEL_PATH,
        "--outdir", OUTDIR
    ]
    print(f"\n[INFO] Lanzando seed {seed} ...")
    subprocess.run(cmd)
    print(f"[INFO] Seed {seed} finalizada.\n")

if __name__ == "__main__":
    seeds = list(range(1, TOTAL_SEEDS + 1))
    print(f"Ejecutando {TOTAL_SEEDS} seeds en batches de {PARALLEL_SEEDS}")

    for i in range(0, TOTAL_SEEDS, PARALLEL_SEEDS):
        batch = seeds[i:i + PARALLEL_SEEDS]
        print(f"\n========== NUEVO BATCH: {batch} ==========")
        with Pool(processes=len(batch)) as pool:
            pool.map(run_seed, batch)
        print(f"========== BATCH {batch} COMPLETADO ==========\n")
        time.sleep(2)

    print("TODAS LAS SEEDS HAN TERMINADO")

end = time.time()
print(f"Total runtime: {(end - start)/60:.2f} minutes")


