# Runs Directory

The cleaned pipeline writes reproducible outputs under:

```text
runs/<run_name>/
  config_used.yaml
  solution_cloud.csv
  reference_front.csv
  archives/
  metrics/
  figures/
  logs/
```

Generated run folders are ignored by Git. Keep only small, intentional reference artifacts in version control.

The legacy `Results/` directory with an uppercase `R` contains historical experiment outputs from the original repository and is preserved for traceability.
