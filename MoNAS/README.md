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


## Dockerfile

The `Dockerfile` is used to create a reproducible execution environment using Docker.

Its main purposes are:
- To ensure that the experiments always run with the same Python version and dependencies.
- To simplify execution on cloud platforms, especially **Microsoft Azure**.
- To avoid dependency and configuration issues when running multiple experiments.

Once the container is built and started, it automatically runs the main experiment script.

## run_all_seeds.py

The `run_all_seeds.py` file is responsible for launching multiple independent experiments using different random seeds.

This script:
- Executes several runs of the main MoNAS algorithm.
- Runs experiments in parallel to reduce total execution time.
- Groups executions to avoid overloading the system.

Each run uses a different seed and stores its results separately, allowing the analysis of variability and consistency across runs.


## Execution Workflow

1. The Docker container prepares the execution environment.
2. `run_all_seeds.py` automatically starts the experiments.
3. Each seed runs the NAS algorithm independently.
4. Results are stored for later analysis.

This setup allows experiments to be executed automatically, reproducibly, and efficiently, both on local machines and cloud infrastructure.

## Research Context

This module supports:
- Large-scale architecture exploration
- Efficient MoNAS under limited computational budgets
- Generation of rich solution clouds for offline archiving
- Controlled and reproducible NAS experiments

## Microsoft Azure Initiative

This work was developed in the context of experiments executed on **Microsoft Azure**, supported by the initiative **Microsoft’s AI for Cultural Heritage**.

This initiative aims to provide cloud computing resources and tools to support research projects that apply artificial intelligence to scientific, cultural, and technological challenges. Azure infrastructure was used to enable scalable and parallel execution of experiments in a reproducible manner.



## Notes

- This module operates **online**, during the NAS search phase.
- It does not perform offline archiving or multi-objective evaluation.
- The implementation is restricted to two objectives, but can be extended to alternative objective formulations.
