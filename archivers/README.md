## Archivers Module

This directory contains the core implementation of the **offline archiving framework** used in the paper.  
It defines the logic for constructing, filtering, truncating, and analyzing Pareto archives from a globally deduplicated solution cloud.

The module is designed around a **central orchestration component** (`brain.py`) that coordinates multiple archiving strategies and a set of helper utilities that ensure deterministic, reproducible, and structurally consistent archive construction.


## Core Design Philosophy


The module separates:
- **Orchestration logic** (what to apply, when, and how),
- **Archiving strategies** (how solutions are selected or removed),
- **Supporting utilities** (data handling, normalization, plotting, and record management).



##  `brain.py` — Central Orchestrator

`brain.py` is the **main entry point** of the offline archiving framework.

It is responsible for:
- Loading and preparing the globally deduplicated solution cloud
- Applying deterministic ordering to ensure reproducibility
- Invoking specific archiving strategies
- Enforcing controlled truncation regimes
- Coordinating evaluation and output generation

All archiving strategies are executed through this module, ensuring that:
- Every method receives identical inputs
- Differences in results are attributable only to the archiving principle
- Archive construction is fully reproducible


##  `strategies/` — Archiving Implementations

The `strategies/` subdirectory contains **independent implementations of offline archiving paradigms**.

Each file implements a specific archiving strategy following a common interface defined in `base.py`.  
Strategies operate exclusively on prepared solution sets and do not perform search or evaluation.

This modular design allows:
- Paradigm-level comparison
- Easy extension with new archivers
- Isolation of representation principles


##  Helper Modules

The remaining files support the archiving process:

- **`base.py`**  
  Defines abstract base classes, interfaces, and shared logic for archiving strategies.

- **`pareto.py`**  
  Implements exact dominance-based filtering (PQ archiver), used to construct the reference Pareto front.

- **`dedub.py`**  
  Removes duplicate architectures based on chromosome identity to produce a unique global solution cloud.

- **`normalize.py`**  
  Applies global objective normalization across the solution cloud.

- **`records.py`**  
  Defines structured data containers for solutions, normalized points, and archives.

- **`io_readers.py`**  
  Handles reading and writing of solution data and archive outputs.

- **`plotting.py`**  
  Provides visualization utilities for Pareto fronts and archive approximations.


## Usage

Offline archiving is typically executed by invoking `brain.py` from an experiment script, configuration file, or batch pipeline.  
All strategies and truncation regimes are applied in a controlled and reproducible manner.



##  Notes

- This module operates **offline**, after the NAS search phase has concluded.
- It is independent of the search algorithm and performance predictors.
- The design emphasizes determinism, modularity, and reproducibility.



