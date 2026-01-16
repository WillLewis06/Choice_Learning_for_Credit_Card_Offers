
# Decisions

This document defines **non-negotiable project decisions** for Part 2 and the bonus tasks.
Codex must treat these as constraints, not suggestions.

Any deviation requires an explicit update to this file.

---

## 1. Repository Ownership

- `choice-learn/` is an upstream dependency and MUST NOT be modified.
- `kif/` is the only repository where new code, tests, configs, and docs may be added.

Invariant:
- All diffs submitted for review originate from `kif/`.

---

## 2. Technology Stack

- Language: Python 3
- Frameworks: TensorFlow, TensorFlow Probability
- Prohibited: PyTorch, JAX, custom training loops outside Keras

Invariant:
- All models must be compatible with `tf.keras.Model.save()` / `load_model()`.

---

## 3. Integration Strategy

- Integrate with `choice-learn` by **importing and extending**, never by copying.
- Prefer subclassing `choice_learn.models.base_model.BaseModel`.

Escalation rule:
- If upstream modification seems necessary, document the reason here before proceeding.

---

## 4. Scope Control (Mission Creep Prevention)

For each required task (Part 2 or bonus):

- Implement exactly:
  1. One model
  2. One synthetic data generator
  3. One evaluation protocol

Explicit non-goals:
- Multiple variants
- Performance tuning
- Hyperparameter optimisation
- General-purpose frameworks

Success criterion:
- Demonstrate correctness and alignment with the theoretical requirement, not maximal performance.

---

## 5. Testing as a First-Class Deliverable

- Tests are mandatory for every new component.
- Test absence is a hard failure.

Required test types:
- Unit tests: mathematical and logical correctness
- Integration tests: end-to-end execution and save/load integrity

Invariant:
- All tests must be deterministic and fast.

---

## 6. Documentation Policy

- Code documentation is minimal and functional.
- Theory, motivation, and discussion live only in report documents.

Invariant:
- Docstrings explain *what* and *how*, never *why*.

---

