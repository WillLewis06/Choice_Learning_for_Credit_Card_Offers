# Conventions

This document defines **coding and testing rules** for all code under `kif/`.

---

## 1. Code Style

- Python 3 only.
- PEP8 formatting.
- Explicit, descriptive names.
- No unused imports or dead code.

---

## 2. Docstrings (Mandatory)

All public classes and functions must include a docstring with:

- One-line summary
- Args
- Returns
- Raises (only if non-trivial)

Constraints:
- No equations
- No citations
- No narrative explanation

---

## 3. Model Contracts

- All models must subclass `BaseModel`.
- Implement:
  - `call()`
  - likelihood or loss computation as required
- Respect masking and batching conventions from `choice-learn`.

---

## 4. Synthetic Data Contracts

- Synthetic datasets must:
  - accept a random seed
  - be deterministic
  - generate minimal datasets suitable for testing

- No network or filesystem dependencies in tests.

---

## 5. Testing Rules

- Use pytest.
- Every new module must have:
  - at least one unit test
  - at least one integration test

Tests must verify:
- output shapes
- probability normalisation
- numerical stability (no NaN / inf)
- correct behaviour in limiting cases

---

## 6. Determinism and Reproducibility

- Set NumPy and TensorFlow seeds explicitly.
- Tests must be reproducible across runs.

---

## 7. Explicit Non-Goals

- No refactoring of `choice-learn`.
- No performance benchmarking.
- No hyperparameter tuning frameworks.
- No notebooks for core logic.

---

