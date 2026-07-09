# Reproduction Guide

## Fresh Clone Smoke Test

```bash
git clone https://github.com/SergioSarmientoRosales/MoNAS-ArchivingStrategies.git
cd MoNAS-ArchivingStrategies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --config configs/toy_example.yaml
pytest tests/
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Full Analysis Template

1. Copy `configs/full_pipeline_template.yaml`.
2. Place or generate the full solution-cloud CSV at the configured `input_path`.
3. Check that objective column names and directions match your data.
4. Run:

```bash
python scripts/run_pipeline.py --config configs/full_pipeline_template.yaml
```

## Legacy Experiments

The original NAS and predictor code remains under `MoNAS/`, with historical seed and result artifacts under `Seeds/` and `Results/`. Those scripts may require the heavier `requirements-legacy.txt` dependencies and machine-specific data paths to be adapted before reuse.
