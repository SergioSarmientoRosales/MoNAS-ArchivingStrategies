# Data Format

The cleaned pipeline reads CSV files. The exact column names are configurable in YAML.

## Required Columns

The default configs expect:

- `architecture_id`: stable architecture identity used for deterministic sorting and default deduplication.
- `chromosome`: architecture encoding. This is recommended for traceability.
- `psnr`: restoration-quality objective. Default direction: maximize.
- `params`: model-size objective. Default direction: minimize.

## Optional Columns

- `model`: predictor or source model name.
- `seed`: experimental seed.
- `generation`: NAS generation.
- `source_file`: original source file path.

## Objective Directions

Directions are declared explicitly:

```yaml
objectives:
  - column: psnr
    direction: maximize
  - column: params
    direction: minimize
```

The pipeline converts every objective to a normalized minimization value:

- For minimized objectives: `(x - min) / (max - min)`.
- For maximized objectives: `(max - x) / (max - min)`.

After normalization, lower is better for every `norm_*` objective.

## Deduplication

Duplicates are detected with `deduplication_key`. The default key is `architecture_id`. If duplicate keys exist, the pipeline keeps the row with the best deterministic normalized objective ranking.

## Missing Values

Rows with missing objective values are dropped during input loading. Missing required columns raise a clear error.

## Output Formats

- `solution_cloud.csv`: validated, normalized, deduplicated solutions.
- `reference_front.csv`: non-dominated subset of the solution cloud.
- `archives/<archiver>_k<size>.csv`: selected approximation archive.
- `metrics/archive_metrics.csv`: indicator table with value and direction.
