# Examples

These examples run on a tiny synthetic objective space and do not require the full experimental data.

## Minimal toy example

```bash
python examples/minimal_toy_example.py
```

The script creates a deterministic synthetic CSV, runs the full pipeline, prints the metric table, and writes outputs under `runs/examples/minimal_toy_example/`.

## Run all archivers on a small input file

```bash
python examples/run_all_archivers_example.py
```

This example uses `examples/example_input.csv`, runs all implemented archivers, and writes outputs under `runs/examples/run_all_archivers/`.

Both examples are intended as smoke tests and API demonstrations. They are not paper-level experiments.
