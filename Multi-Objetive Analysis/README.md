# Multi-Objective Analysis Module

This directory contains the implementation of the **offline multi-objective evaluation framework** used to assess the quality of Pareto front approximations produced by different archiving strategies.

The module provides a structured and reproducible pipeline for computing convergence, coverage, preference representativeness, approximation bounds, and worst-case deviation metrics with respect to a reference Pareto front.


## Core Design Philosophy

Each indicator captures a complementary structural aspect of archive quality, enabling principled comparison of different Pareto representations under controlled compression regimes.

The module is organized around:
- A **central orchestration component** that coordinates metric computation,
- **Individual indicator implementations** with clear semantic roles,
- **Helper utilities** for data handling, visualization, and reporting.


---

##  `metrics_brain.py` — Central Orchestrator

`metrics_brain.py` acts as the **main entry point** for multi-objective evaluation.

It is responsible for:
- Loading archived solution sets and the reference Pareto front
- Applying consistent normalization and preprocessing
- Invoking individual performance indicators
- Aggregating metric results across truncation regimes and strategies
- Producing structured outputs for analysis and visualization

All evaluation runs are executed through this module, ensuring:
- Identical evaluation conditions across archivers
- Consistent metric definitions
- Reproducible quantitative comparisons

---

##  Performance Indicators Implemented

The module includes complementary indicators covering multiple aspects of archive quality:

- **IGD+ (`igdplus_metric.py`)**  
  Measures dominance-consistent convergence and coverage relative to the reference front.

- **Hypervolume (`hv2d.py`)**  
  Evaluates global dominance quality and dominated-space coverage in objective space.

- **R2 Indicator (`r2_indicator.py`)**  
  Assesses preference-space representativeness using multiple scalar utility functions.

- **ε-indicator (`epsilon_indicator.py`)**  
  Quantifies approximation bounds and tolerance-based dominance.

- **Hausdorff Distance (`hausdorff_metric.py`)**  
  Captures worst-case geometric deviation between approximation and reference fronts.

- **Coverage Metrics (`coverage_metric.py`)**  
  Provide dominance-based coverage analysis.

## Helper Modules


- **`io_fronts.py`**  
  Handles loading, saving, and formatting of Pareto fronts and archive representations.

- **`plotting_cloud.py`**  
  Visualizes the global solution cloud and reference geometry.

- **`plots.py`**  
  Generates comparative visualizations of archived fronts across regimes.

- **`report.py`**  
  Aggregates metrics, formats results, and produces structured summaries for analysis.


## Usage

Multi-objective evaluation is typically executed by invoking `metrics_brain.py` after offline archiving has been completed.  
The module assumes:
- A fixed reference Pareto front
- Deterministic archive construction
- Controlled truncation regimes


##  Research Context

This module supports:
- Quantitative evaluation of Pareto front approximations
- Structural comparison across archiving paradigms
- Regime-dependent compression analysis
- Reproducible multi-objective performance assessment


## Notes

- All metrics are computed **offline**, after NAS search and archiving.
- The module is independent of the NAS search algorithm.
- Indicator implementations are modular and extensible.


