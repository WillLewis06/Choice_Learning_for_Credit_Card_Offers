# Architecture

This document defines **exact module boundaries** and **extension points**.
Codex must not infer structure beyond what is written here.

---

## 1. Dependency Boundary

- `choice-learn/` is a read-only dependency.
- All new functionality lives under `kif/`.
- `kif/` imports from `choice_learn`, never the reverse.

---

## 2. kif Directory Structure

Mandatory top-level layout:

- `kif/src/`
  Python package containing all implementations.

- `kif/tests/`
  Pytest unit and integration tests.

- `kif/configs/`
  YAML configs for synthetic experiments.

- `kif/docs/`
  Planning documents and reports.

---

## 3. Module Placement (Strict)

### Part 2: Lu-style model

- Model:
  - `kif/src/kif_models/lu25_model.py`
- Synthetic dataset:
  - `kif/src/kif_datasets/synthetic_lu25.py`

Responsibilities:
- Implement Lu-style assumptions.
- Remain compatible with `choice_learn` training and evaluation flows.

---

### Part 2 Extension: Zhang + Lu Hybrid

- Model:
  - `kif/src/kif_models/deephalo_lu25.py`
- Synthetic dataset:
  - `kif/src/kif_datasets/synthetic_deephalo_lu25.py`

Responsibilities:
- Extend Halo-style utilities with unobserved market–product shocks.
- Preserve baseline behaviour in limiting cases.

---

### Bonus 1: Dynamic Inventory (Ching-style)

- Model:
  - `kif/src/kif_models/dynamic_inventory.py`
- Synthetic dataset:
  - `kif/src/kif_datasets/synthetic_inventory.py`

Constraints:
- Small, explicit state space.
- Direct Bellman recursion; no general DP solvers.

---

### Bonus 2: Habit Formation and Peer Effects

- Model:
  - `kif/src/kif_models/habit_peer.py`
- Synthetic dataset:
  - `kif/src/kif_datasets/synthetic_habit_peer.py`

Constraints:
- One habit mechanism.
- One peer mechanism.
- No network inference.

---

### Bonus 3: Constrained Assortment Optimisation

- Optimiser:
  - `kif/src/kif_toolbox/constrained_assortment.py`

Constraints:
- Use probabilities from a trained choice model.
- Support only:
  - must-link constraints
  - cannot-link constraints

---

## 4. Test Placement

- Unit tests:
  - `kif/tests/unit/models/`
  - `kif/tests/unit/datasets/`
  - `kif/tests/unit/toolbox/`

- Integration tests:
  - `kif/tests/integration/`

Integration tests must:
- run full pipelines on small synthetic data
- validate save/load and determinism

---

