# Data Directory

Place external or reconstructed input data here when running the full pipeline.

The cleaned pipeline expects a CSV file with one row per candidate architecture. Required columns are configured in YAML, but the default template expects:

| Column | Required | Meaning |
| --- | --- | --- |
| `architecture_id` | yes | Stable unique identifier for an architecture. |
| `chromosome` | recommended | Architecture encoding used for deduplication or traceability. |
| `psnr` | yes | Restoration quality objective, maximized by default. |
| `params` | yes | Model-size objective, minimized by default. |
| `model` | optional | Predictor/model source. |
| `seed` | optional | Random seed or experimental replicate. |
| `generation` | optional | NAS generation or iteration. |

Large raw datasets and generated solution clouds should not be committed unless they are intentionally part of the research artifact. Use `examples/example_input.csv` for a small runnable example.
