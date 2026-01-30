# MoNAS Module

This directory contains the implementation of the **multi-objective neural architecture search (MoNAS)** pipeline used in the study.  
It integrates a two-objective evolutionary search algorithm with model-based performance predictors to guide exploration and generate large-scale solution sets for offline archiving and analysis.

## Core Design Philosophy

The MoNAS pipeline is designed to decouple:
- **Search dynamics**
- **Performance estimation**

This separation allows the search phase to prioritize efficient exploration.


## `nsga3_model_based_2obj_compasr.py` — MoNAS Search Algorithm

This script implements the **two-objective MoNAS algorithm** based on **NSGA-III**.

Key characteristics:
- Joint optimization of **reconstruction quality** (PSNR, predicted) and **model complexity** (number of parameters)
- Predictor-guided evaluation of candidate architectures
- Population-based evolutionary search
- Generation of large and diverse sets of candidate architectures
- Designed for large-scale exploration rather than final model selection

The output of this script is a collection of evaluated architectures that form the basis of the global solution cloud.


## `Model_Based_Predictor/` — Performance Prediction Framework

This directory contains all components related to **model-based performance estimation**, used during the search phase to avoid full training of candidate architectures.

### `Hyperparameters_Search/`

Contains:
- Code for hyperparameter optimization of performance predictors
- The architecture–performance dataset used for training and validation
- Bayesian optimization and cross-validation routines


### `Model_Based_C_index/`

Contains:
- Final trained predictor models optimized to **maximize the C-index**
- Models focused on preserving the **relative ranking** of architectures
- Used during MoNAS to guide efficient exploration of the search space

These predictors prioritize correct ordering of architectures over absolute prediction accuracy.


### `Model_Based_MSE/`

Contains:
- Final trained predictor models optimized to **minimize MSE**
- Models focused on **numerical fidelity** of PSNR estimation
- Used for post-search re-evaluation and normalization of the global solution cloud

These predictors provide a consistent error model for offline analysis and fair comparison across archiving strategies.

## Search–Evaluation Separation

During MoNAS:
- **C-index–optimized predictors** guide the evolutionary search
- Architecture exploration is prioritized over exact performance estimation

After MoNAS:
- **MSE-optimized predictors** re-evaluate all discovered architectures
- A homogeneous prediction space is established for offline archiving and analysis

This two-stage predictor strategy decouples exploration efficiency from representation accuracy.


## Research Context

This module supports:
- Large-scale architecture exploration
- Efficient MoNAS under limited computational budgets
- Generation of rich solution clouds for offline archiving
- Controlled and reproducible NAS experiments


## Notes

- This module operates **online**, during the NAS search phase.
- It does not perform offline archiving or multi-objective evaluation.
- The implementation is restricted to two objectives, but can be extended to alternative objective formulations.
