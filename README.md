# JPM-Q3

Implementation of the JPM-Q3 modelling stack (see report for full context). This repository includes:
- Zhang: context-dependent discrete choice models (featureless, feature-based, stacked)
- Lu: market–product shock recovery (BLP benchmarks + Bayesian shrinkage)
- Ching: stockpiling dynamics estimated via Bayesian MCMC
- Bonus2: habit/peer/seasonality dynamics estimated via Bayesian MCMC
- Synthetic DGPs and pytest tests

## Install (repo root)

    pip install -U pip
    pip install -e .


## Repo structure

- `zhang/`  
  Context-dependent choice models in the style of Zhang. Contains the core Zhang choice learn model. It includes the featureless and feature-based variants and a stacked variant used for deeper interaction modelling. Shared building blocks also live here.

- `lu/`  
  Market–product shock modelling fro mteh Lu paper.
  - BLP-style benchmarks for demand-shock recovery, including inversion utilities and instrument variants
  - A Bayesian shrinkage estimator for structured market and product shocks
  - A choice-learn style implementation that aligns the shrinkage workflow with the choice-model outputs

- `ching/`  
  Stockpiling take-up dynamics and Bayesian estimation.
  - Model and state evolution used to generate utilities over time
  - Posterior definitions and Metropolis–Hastings updates
  - Diagnostics and evaluation utilities
  - Input validation so simulation and estimation fail fast on shape and configuration issues

- `bonus2/`  
  Dynamic extensions for habit formation, peer exposure, and seasonality, estimated via Bayesian MCMC. Mirrors the structure of `ching/`:
  - Model and state evolution for habit, peer, and time effects
  - Posterior definitions and Metropolis–Hastings updates
  - Diagnostics and evaluation utilities
  - Input validation for shapes and configuration integrity

- `datasets/`  
  Synthetic DGP generators used by the run scripts. Produces artificial datasets for each experiment

- `toolbox/`  
  - Reusable MCMC kernel implementations
  - Market shock estimator assessment tool

- `tests/`  
  Pytest suite that checks DGP schemas, validates core model components, and exercises estimator workflows. The intent is to cover the main interfaces that the run scripts depend on.

- `zhang_pipeline/`  
  Standalone config-driven training and evaluation pipeline for Zhang models. This is separate from the synthetic experiments and is intended for training on CSV datasets.
  - `configs/` contains YAML configs for train, continue-train, and evaluate runs
  - `data/` is where input CSV datasets must be placed
  - `checkpoints/` stores training outputs and saved models
  - `models/`, `support/`, and `training/` contain the pipeline internals and utilities

## What runs what

There are two usage modes.

### A) Synthetic experiments (paper-style / report experiments)

Run from the repo root. Each script generates synthetic data via `datasets/`, runs estimation, and prints diagnostics/results.

1) `run_zhang_with_lu.py`
- End-to-end integration experiment:
  - trains a Zhang choice model on synthetic choice data
  - adds Lu-style market/product shocks
  - evaluates shock recovery using Lu estimators

2) `run_lu.py`
- Lu-style simulation harness:
  - generates Lu DGPs (markets/products/shocks)
  - runs BLP benchmarks (strong/weak IV variants) and the shrinkage estimator
  - prints recovery diagnostics (e.g., RMSE/correlation-style summaries)

3) `run_ching.py`
- Ching stockpiling estimation:
  - generates a stockpiling panel
  - estimates stockpiling parameters via MCMC
  - prints fit + parameter recovery diagnostics

4) `run_bonus2.py`
- Bonus2 dynamic effects estimation:
  - generates a habit/peer/seasonality panel
  - estimates dynamic parameters via MCMC
  - prints fit + parameter recovery diagnostics

5) `run_cl_models.py`
- Benchmark comparison runner:
  - runs comparisons across the choice-learn style baselines/variants used in the project


### B) Standalone Zhang training pipeline (config-driven)

1) Put dataset CSV files into:

    zhang_pipeline/data/

2) Run entrypoints using `--config` (paths relative to `zhang_pipeline/`):

    cd zhang_pipeline
    python train.py --config configs/<train_config>.yaml
    python continue_train.py --config configs/<continue_train_config>.yaml
    python evaluate.py --config configs/<evaluate_config>.yaml

## Tests

Run from the repo root:

    pytest
