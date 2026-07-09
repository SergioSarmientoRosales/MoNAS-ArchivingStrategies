# MoNAS-ArchivingStrategies

Reproducible offline analysis framework for Pareto archiving strategies in multi-objective neural architecture search (MoNAS) for super-resolution image restoration.

This repository supports the work:

**An Offline Analysis of Pareto Archiving Strategies in Multiobjective Neural Architecture Search for Super-Resolution Image Restoration**
Sarmiento-Rosales, Sergio et al. Journal/status: TBD, 2026.

The repository now includes a clean, lightweight Python pipeline that an external researcher can run from a fresh clone, plus the original historical MoNAS, seed, model, and result artifacts for traceability.

## Scientific Motivation

Multi-objective NAS often produces large sets of candidate architectures. Even after identifying non-dominated architectures, retraining or inspecting every candidate can be expensive. Offline Pareto archiving studies how to represent a large solution cloud with a smaller archive while preserving useful front structure.

This project treats archiving as a reproducible representation problem:

1. Build or load a global solution cloud.
2. Deduplicate architectures.
3. Normalize objectives with explicit directions.
4. Construct an empirical reference Pareto front.
5. Apply archiving and truncation strategies.
6. Evaluate approximations with multi-objective indicators.
7. Save tables and plots for inspection.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `src/monas_archiving/` | Clean package for data validation, normalization, Pareto utilities, archivers, indicators, plotting, and pipeline execution. |
| `configs/` | YAML configs for the toy example and a full-pipeline template. |
| `scripts/` | Command-line entry points for each pipeline stage and one unified runner. |
| `examples/` | Small runnable examples and toy input data. |
| `docs/` | Audit notes, data format, pipeline guide, archiver guide, indicator guide, and reproduction guide. |
| `tests/` | Lightweight pytest suite for core logic and smoke execution. |
| `data/` | Placeholder for external full input data. |
| `runs/` | Placeholder for generated outputs from the cleaned pipeline. |
| `archivers/` | Original legacy archiving scripts, preserved for traceability. |
| `Multi-Objetive Analysis/` | Original legacy metric/plotting scripts. |
| `MoNAS/` | Original NAS and predictor-training scripts. |
| `Seeds/` | Historical seed-level raw and processed CSV artifacts. |
| `Results/` | Historical result artifacts from the original repository. |
| `requirements.txt` | Lightweight dependencies for the cleaned pipeline, tests, examples, and CI. |
| `requirements-legacy.txt` | Original heavier dependency snapshot for legacy NAS/predictor scripts. |

## Installation

```bash
git clone https://github.com/SergioSarmientoRosales/MoNAS-ArchivingStrategies.git
cd MoNAS-ArchivingStrategies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The cleaned pipeline does not require TensorFlow, XGBoost, or GPU access. Those heavier dependencies are only needed for some legacy NAS/predictor scripts and are listed in `requirements-legacy.txt`.

## Quickstart

Run the full toy pipeline:

```bash
python scripts/run_pipeline.py --config configs/toy_example.yaml
```

Expected outputs:

```text
runs/toy_example/
  config_used.yaml
  solution_cloud.csv
  reference_front.csv
  archives/
  metrics/archive_metrics.csv
  figures/
```

Run the minimal example script:

```bash
python examples/minimal_toy_example.py
```

Run all tests:

```bash
pytest tests/
```

## Step-by-Step Pipeline

Each stage can be run independently:

```bash
python scripts/01_build_solution_cloud.py --config configs/toy_example.yaml
python scripts/02_build_reference_front.py --config configs/toy_example.yaml
python scripts/03_run_archivers.py --config configs/toy_example.yaml
python scripts/04_evaluate_archives.py --config configs/toy_example.yaml
python scripts/05_generate_plots.py --config configs/toy_example.yaml
```

Or run everything:

```bash
python scripts/run_pipeline.py --config configs/toy_example.yaml
```

## Input Data Format

The default configs expect a CSV with:

| Column | Required | Default direction | Meaning |
| --- | --- | --- | --- |
| `architecture_id` | yes | n/a | Stable architecture identifier. |
| `chromosome` | recommended | n/a | Architecture encoding for traceability. |
| `psnr` | yes | maximize | Restoration quality objective. |
| `params` | yes | minimize | Model-size objective. |
| `model` | optional | n/a | Predictor or source model. |
| `seed` | optional | n/a | Experimental seed. |
| `generation` | optional | n/a | NAS generation. |

Objective directions are explicit in YAML:

```yaml
objectives:
  - column: psnr
    direction: maximize
  - column: params
    direction: minimize
```

The pipeline converts every objective to normalized minimization form. After normalization, lower values are better for all `norm_*` columns.

See [docs/data_format.md](docs/data_format.md) for full details.

## Implemented Archivers

The cleaned pipeline implements deterministic, lightweight versions of:

- `pq`: exact non-dominated archive with crowding truncation.
- `hv`: greedy hypervolume-oriented archive.
- `r2`: greedy R2-oriented archive.
- `crowding`: crowding-distance truncation.
- `grid`: grid-based archiving.
- `epsilon`: epsilon-box archiving.
- `tight1`: structure-preserving farthest-point selection.
- `kmeans`: seeded k-means representative selection.
- `entropy`: grid-rarity selection.

See [docs/archivers.md](docs/archivers.md).

## Evaluation Indicators

The cleaned pipeline reports:

| Indicator | Direction |
| --- | --- |
| `igd_plus` | lower is better |
| `hypervolume` | higher is better |
| `r2` | lower is better |
| `epsilon` | lower is better |
| `hausdorff` | lower is better |

See [docs/indicators.md](docs/indicators.md).

## Full Reproduction

Use `configs/full_pipeline_template.yaml` as a starting point:

1. Copy the template.
2. Place or generate the full solution-cloud CSV at the configured `input_path`.
3. Confirm objective columns, directions, deduplication key, archivers, and truncation sizes.
4. Run:

```bash
python scripts/run_pipeline.py --config configs/full_pipeline_template.yaml
```

The historical `Seeds/` and `Results/` directories are preserved, but the clean pipeline expects an explicit solution-cloud CSV. If you reconstruct the cloud from historical seed files, save it as a CSV matching the documented format.

## Troubleshooting

`ModuleNotFoundError: monas_archiving`

Run `pip install -e .`, or execute the scripts from the repository root.

`Input CSV is missing required columns`

Check the config column names against your CSV. Required columns are defined by `architecture_id_column`, `chromosome_column`, and `objectives`.

`Hypervolume supports exactly two objectives`

The current cleaned implementation supports two-objective hypervolume because the project focuses on PSNR and parameter count.

Legacy scripts fail because of local paths

Some original scripts contain machine-specific paths from the historical research environment. Prefer the cleaned `scripts/` pipeline for new runs, or adapt the legacy paths manually.

## Citation

```bibtex
@article{SarmientoRosales2026OfflineArchiving,
  title = {An Offline Analysis of Pareto Archiving Strategies in Multiobjective Neural Architecture Search for Super-Resolution Image Restoration},
  author = {Sarmiento-Rosales, Sergio and others},
  journal = {TBD},
  year = {2026}
}
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).

## Contact

Open an issue on GitHub for questions, reproduction problems, or documentation gaps:

https://github.com/SergioSarmientoRosales/MoNAS-ArchivingStrategies/issues
