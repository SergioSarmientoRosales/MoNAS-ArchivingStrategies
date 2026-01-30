# MoNAS-ArchivingStrategies
Reproducible framework for offline analysis of Pareto archiving strategies in MoNAS. Includes global solution cloud construction, reference front generation, controlled compression, and structural evaluation of representations. Enables compact Pareto approximations and reduces the number of architectures requiring full retraining.


This repository contains the code and experimental framework for the paper:

**“An Offline Analysis of Pareto Archiving Strategies in Multiobjective Neural Architecture Search for Super-Resolution Image Restoration”**

The project provides a large-scale, reproducible framework for analyzing offline archiving strategies in multi-objective neural architecture search (MoNAS), with a focus on representation, compression, and structural approximation of Pareto fronts in super-resolution image restoration (SRIR).

## Overview

Multi-objective NAS generates large sets of candidate architectures and Pareto-optimal solutions.  
Even when a Pareto front is obtained, retraining all non-dominated architectures remains computationally infeasible in large-scale settings.

This repository implements an **offline archiving framework** that:

- Constructs a **globally deduplicated solution cloud**
- Builds a **reference Pareto front**
- Applies multiple **offline archiving strategies** as controlled approximations
- Evaluates compression under multiple **truncation regimes**
- Enables **compact Pareto representations**, reducing the number of architectures that require full retraining
- Supports **reproducible, paradigm-level comparison** of archiving strategies

Offline archiving is treated as a **representation and compression problem**, not merely a selection mechanism.

## Archiving Paradigms Implemented

The framework includes representative strategies from multiple paradigms:

- **Exact dominance:** PQ
- **Geometric quality indicators:** Hypervolume (HV)
- **Preference-space indicators:** R2
- **Local density regulation:** Crowding Distance
- **Spatial discretization:** Grid-based archiving
- **Tolerance-based discretization:** Epsilon-dominance (Eps1)
- **Structure-preserving strategies:** Tight1
- **Geometric clustering:** K-means
- **Information-theoretic selection:** Entropy-based archiving

Each strategy is treated as a **representation model** for Pareto front approximation.

## Core Pipeline

1. **Search phase**
   - Multi-objective NAS using NSGA-III
   - Predictor-guided evaluation
   - Large-scale architecture exploration

2. **Global solution cloud construction**
   - Aggregation across predictors and random seeds
   - Deduplication by chromosome identity
   - Global normalization

3. **Reference front construction**
   - Exact dominance filtering (PQ archiver)
   - Reference Pareto front used as empirical ground truth

4. **Offline archiving**
   - Controlled application of archivers
   - Deterministic input ordering
   - Paradigm-level representation

5. **Controlled truncation**
   - Relative archive sizes: 75%, 50%, 25%
   - Structural compression regimes

6. **Evaluation**
   - IGD+, HV, R2, ε-indicator, Hausdorff distance
   - Structural and geometric analysis
   - Quantitative comparison

## Evaluation Indicators

- **IGD+** — convergence and coverage consistency  
- **Hypervolume (HV)** — global dominance quality  
- **R2** — preference-space representativeness  
- **ε-indicator** — approximation bounds  
- **Hausdorff distance** — worst-case geometric deviation  


