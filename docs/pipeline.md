# Pipeline

The cleaned pipeline makes the offline analysis order explicit.

## Step-by-Step Commands

```bash
python scripts/01_build_solution_cloud.py --config configs/toy_example.yaml
python scripts/02_build_reference_front.py --config configs/toy_example.yaml
python scripts/03_run_archivers.py --config configs/toy_example.yaml
python scripts/04_evaluate_archives.py --config configs/toy_example.yaml
python scripts/05_generate_plots.py --config configs/toy_example.yaml
```

The same pipeline can be run in one command:

```bash
python scripts/run_pipeline.py --config configs/toy_example.yaml
```

## Conceptual Stages

1. Load or construct the global solution cloud.
2. Validate required columns and objective values.
3. Normalize objectives with explicit directions.
4. Deduplicate architectures.
5. Construct the empirical reference Pareto front.
6. Apply offline archiving strategies.
7. Apply truncation sizes.
8. Evaluate archives with indicators.
9. Generate lightweight plots.
10. Save all outputs under `runs/<run_name>/`.

## Output Tree

```text
runs/<run_name>/
  config_used.yaml
  solution_cloud.csv
  reference_front.csv
  archives/
  metrics/archive_metrics.csv
  figures/
  logs/
```
