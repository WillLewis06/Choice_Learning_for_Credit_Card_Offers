"""
End-to-end pytest tests for evaluate.py.

These tests check that:
- evaluate.py can be executed using only a .yaml config file
- a previously saved .keras model can be loaded correctly
- evaluation runs to completion on valid data
- invalid inputs get caught early

The tests treat evaluate.py as a CLI entrypoint with the config file passed as an option
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    """
    Returns the repo path: the parent dir of the test/ dir.
    """
    return Path(__file__).resolve().parents[1]


def _train_py() -> Path:
    """
    Checks train.py exists and returns its path.
    """
    p = _repo_root() / "train.py"
    if not p.exists():
        raise RuntimeError(f"Could not find train.py at: {p}")
    return p


def _evaluate_py() -> Path:
    """
    Checks evaluate.py exists and returns its path.
    """
    p = _repo_root() / "evaluate.py"
    if not p.exists():
        raise RuntimeError(f"Could not find evaluate.py at: {p}")
    return p


def _run_train(cfg_path: Path) -> subprocess.CompletedProcess[str]:
    """
    Executes train.py as a subprocess using a given config file.
    Enforces CPU for simplicity.
    We need to train a model before we can evaluate it.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    return subprocess.run(
        ["python", str(_train_py()), "--config", str(cfg_path)],
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
    )


def _run_evaluate(cfg_path: Path) -> subprocess.CompletedProcess[str]:
    """
    Executes evaluate.py as a subprocess using a given config file.
    Enforces CPU for simplicity.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    return subprocess.run(
        ["python", str(_evaluate_py()), "--config", str(cfg_path)],
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
    )


def _latest_keras(checkpoint_dir: Path) -> Path:
    """
    Return the latest saved .keras model file in a checkpoint directory.
    """
    saved = sorted(checkpoint_dir.glob("*.keras"), key=lambda p: p.stat().st_mtime)
    assert saved, f"No .keras models found in {checkpoint_dir}"
    return saved[-1]


def _write_featureless_csv(
    path: Path,
    n_rows: int,
    num_items: int,
) -> None:
    """
    Create a tiny synthetic featureless dataset for testing.

    The generated data satisfies all constraints enforced by validate_raw.
    """
    rows = []

    for i in range(n_rows):
        offered = ([1, 0, 1, 0] if i % 2 == 0 else [1, 1, 0, 0])[:num_items]
        if sum(offered) == 0:
            offered[0] = 1

        chosen_idx = next(j for j, v in enumerate(offered) if v == 1)

        y = [0] * num_items
        y[chosen_idx] = 1

        row = {f"X{j+1}": offered[j] for j in range(num_items)}
        row.update({f"Y{j+1}": y[j] for j in range(num_items)})
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def _write_yaml(path: Path, text: str) -> None:
    """
    Saves a config to a .yaml file.
    """
    path.write_text(text.lstrip(), encoding="utf-8")


def test_evaluate_featureless_runs(tmp_path: Path) -> None:
    """
    Run evaluate.py end-to-end on a trained featureless model.

    Expected behaviour:
    - a model trained by train.py is loaded
    - evaluation completes successfully
    - a finite evaluation metric is reported
    """

    # Create tmp directories and write synthetic CSV data
    data_root = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    data_root.mkdir()
    checkpoint_dir.mkdir()

    _write_featureless_csv(data_root / "train.csv", n_rows=12, num_items=4)
    _write_featureless_csv(data_root / "test.csv", n_rows=6, num_items=4)

    # Train a base model
    train_cfg = tmp_path / "cfg_train.yaml"
    _write_yaml(
        train_cfg,
        f"""
data_root: "{data_root}"
checkpoint_dir: "{checkpoint_dir}"

files:
  train_raw: "train.csv"
  test_raw: "test.csv"

model:
  type: "featureless"
  depth: 2
  width: 8

train:
  loss: "nll"
  optimizer: "adam"
  lr: 0.001
  batch_size: 4
  epochs: 1

eval:
  batch_size: 32
  loss: "nll"
""",
    )

    proc_train = _run_train(train_cfg)
    assert proc_train.returncode == 0

    model_path = _latest_keras(checkpoint_dir)

    # Evaluate the model
    eval_cfg = tmp_path / "cfg_eval.yaml"
    _write_yaml(
        eval_cfg,
        f"""
data_root: "{data_root}"

files:
  train_raw: "train.csv"
  test_raw: "test.csv"

model:
  type: "featureless"
  load_path: "{model_path}"

eval:
  batch_size: 32
  loss: "nll"
""",
    )

    proc_eval = _run_evaluate(eval_cfg)
    assert (
        proc_eval.returncode == 0
    ), f"evaluate.py failed.\nstdout:\n{proc_eval.stdout}\n\nstderr:\n{proc_eval.stderr}"

    # Check a finite metric was printed
    m = re.search(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", proc_eval.stdout)
    assert m, f"No metric found in stdout:\n{proc_eval.stdout}"
    val = float(m.group(1))
    assert math.isfinite(val)


def test_evaluate_fails_missing_model(tmp_path: Path) -> None:
    """
    Check evaluate.py fails fast when model.load_path does not exist.
    """

    data_root = tmp_path / "data"
    data_root.mkdir()

    _write_featureless_csv(data_root / "train.csv", n_rows=4, num_items=4)
    _write_featureless_csv(data_root / "test.csv", n_rows=4, num_items=4)

    cfg_path = tmp_path / "cfg_bad_eval.yaml"
    _write_yaml(
        cfg_path,
        f"""
data_root: "{data_root}"

files:
  train_raw: "train.csv"
  test_raw: "test.csv"

model:
  type: "featureless"
  load_path: "{tmp_path / 'missing.keras'}"

eval:
  batch_size: 32
  loss: "nll"
""",
    )

    proc = _run_evaluate(cfg_path)
    assert proc.returncode != 0
    combined = (proc.stdout + "\n" + proc.stderr).lower()
    assert "load" in combined or "not found" in combined or "exist" in combined
