# Repository Audit

## Current State

The original repository contains a mixture of source code, seed-level CSV data, processed CSV data, historical result folders, trained predictor models stored with Git LFS, and analysis scripts.

Top-level legacy folders observed during audit:

- `archivers/`: original archiving implementations and a hard-coded driver script.
- `Multi-Objetive Analysis/`: metric and plotting scripts; the directory name contains a space and a typo.
- `MoNAS/`: NSGA-III and predictor-training scripts for the original NAS workflow.
- `Seeds/`: raw and processed seed CSVs.
- `Results/`: historical global Pareto archive outputs.

## Main Problems Found

- Several scripts contain hard-coded machine-specific paths such as `G:\Mi unidad\Paper2\Seeds`.
- Some original `archivers/` modules use fragile imports such as `from records import ...` and do not include `__init__.py`, making clean-clone imports unreliable.
- The original root `requirements.txt` mixed lightweight analysis dependencies with heavy NAS/predictor dependencies such as TensorFlow and XGBoost.
- Reproduction order was implicit and split across multiple folders.
- The repository tracked many historical outputs but did not provide a small example that runs end-to-end.
- Data format assumptions were partly encoded in scripts rather than documented centrally.

## Implemented Cleanup Strategy

The cleanup adds a new `src/monas_archiving/` package with a deterministic offline analysis pipeline. Legacy files are preserved rather than deleted, because they contain historical experimental context and trained model artifacts.

The new package focuses on:

1. Input validation.
2. Explicit objective directions.
3. Direction-aware min-max normalization.
4. Deduplication.
5. Empirical reference front construction.
6. Deterministic archiving strategies.
7. Indicator evaluation.
8. Reproducible outputs and plots.

This preserves scientific intent while making a clean-clone path available for external users.
