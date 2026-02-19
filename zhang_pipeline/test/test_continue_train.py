"""
End-to-end pytest tests for continue_train.py.

These tests check that:
- continue_train.py can be executed using only a .yaml config file
- a previously saved .keras model can be loaded correctly
- continued training runs to completion and saves a model
- invalid inputs get caught early

The tests treat continue_train.py as a CLI entrypoint with the config file passed as an option
"""

from __future__ import annotations

import os
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


def _continue_train_py() -> Path:
    """
    Checks continue_train.py exists and returns its path.
    """
    p = _repo_root() / "continue_train.py"
    if not p.exists():
        raise RuntimeError(f"Could not find continue_train.py at: {p}")
    return p


def _run_train(cfg_path: Path) -> subprocess.CompletedProcess[str]:
    """
    Executes train.py as a subprocess using a given config file.
    Enforces CPU for simplicity.
    We need to train a model to be able to test the continue train part
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


def _run_continue_train(cfg_path: Path) -> subprocess.CompletedProcess[str]:
    """
    Executes continue_train.py as a subprocess using a given config file.
    Enforces CPU for simplicity.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    return subprocess.run(
        ["python", str(_continue_train_py()), "--config", str(cfg_path)],
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
    )


def _write_featureless_csv(path: Path, n_rows: int, num_items: int) -> None:
    """
    Create a tiny synthetic featureless dataset for testing.

    The generated data satisfies all constraints enforced by validate_raw:
    - X entries are binary (0/1)
    - each row has at least one offered item
    - Y is one-hot
    - Y is zero where X is zero

    Saves as .csv
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


def _write_featurebased_csv(
    path: Path,
    *,
    group_col: str,
    choice_col: str,
    n_groups: int,
    alts_per_group: list[int],
) -> None:
    """
    Creates a tiny synthetic featurebased dataset for testing.

    Each group has exactly one chosen alternative and multiple numeric features.
    This is sufficient to exercise grouping, padding, and masking logic.

    Saves as .csv
    """
    rows = []

    for g in range(n_groups):
        num_alts = alts_per_group[g]
        chosen_alt = 0

        for a in range(num_alts):
            rows.append(
                {
                    group_col: g,
                    "price": float(10 + g + a),
                    "quality": float(0.5 + 0.1 * ((g + a) % 3)),
                    choice_col: 1 if a == chosen_alt else 0,
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def _write_yaml(path: Path, text: str) -> None:
    """
    Saves a config to a .yaml file.
    """
    path.write_text(text.lstrip(), encoding="utf-8")


def _latest_keras(checkpoint_dir: Path) -> Path:
    """
    Return the latest saved .keras model file in a checkpoint directory.
    """
    saved = sorted(checkpoint_dir.glob("*.keras"), key=lambda p: p.stat().st_mtime)
    assert saved, f"No .keras models found in {checkpoint_dir}"
    return saved[-1]


def test_continue_train_featureless_loads_and_saves(tmp_path: Path) -> None:
    """
    Run continue_train.py end-to-end using featureless data.

    Expected behaviour:
    - train.py produces a saved .keras model
    - continue_train.py loads that model
    - continued training completes successfully
    - a new model file is saved
    """

    # Create tmp directories and write synthetic CSV data
    data_root = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    data_root.mkdir()
    checkpoint_dir.mkdir()

    _write_featureless_csv(data_root / "train.csv", n_rows=12, num_items=4)
    _write_featureless_csv(data_root / "test.csv", n_rows=6, num_items=4)

    # Create tmp config file for train.py
    train_cfg_path = tmp_path / "cfg_train_featureless.yaml"
    _write_yaml(
        train_cfg_path,
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
  patience: null
  l2: 0.0

eval:
  batch_size: 32
  loss: "nll"
""",
    )

    proc_train = _run_train(train_cfg_path)
    assert (
        proc_train.returncode == 0
    ), f"train.py failed.\nstdout:\n{proc_train.stdout}\n\nstderr:\n{proc_train.stderr}"

    base_model_path = _latest_keras(checkpoint_dir)

    # Create tmp config file for continue_train.py
    cont_cfg_path = tmp_path / "cfg_continue_featureless.yaml"
    _write_yaml(
        cont_cfg_path,
        f"""
data_root: "{data_root}"
checkpoint_dir: "{checkpoint_dir}"

files:
  train_raw: "train.csv"
  test_raw: "test.csv"

model:
  type: "featureless"
  load_path: "{base_model_path}"
  save_name: "continued_featureless"

train:
  loss: "nll"
  optimizer: "adam"
  lr: 0.001
  batch_size: 4
  epochs: 1
  patience: null
  l2: 0.0

eval:
  batch_size: 32
  loss: "nll"
""",
    )

    proc_cont = _run_continue_train(cont_cfg_path)
    assert proc_cont.returncode == 0, (
        "continue_train.py failed.\n"
        f"stdout:\n{proc_cont.stdout}\n\nstderr:\n{proc_cont.stderr}"
    )

    # Load check: confirm the continue run reported a model load
    combined = (proc_cont.stdout + "\n" + proc_cont.stderr).lower()
    assert "load" in combined or "loaded" in combined, (
        "Expected continue_train.py to report loading a model.\n"
        f"stdout:\n{proc_cont.stdout}\n\nstderr:\n{proc_cont.stderr}"
    )

    saved_after = list(checkpoint_dir.glob("*.keras"))
    assert len(saved_after) >= 2, (
        "Expected at least two saved models (base + continued).\n"
        f"stdout:\n{proc_cont.stdout}\n\nstderr:\n{proc_cont.stderr}"
    )


def test_continue_train_featurebased_loads_and_saves(tmp_path: Path) -> None:
    """
    Run continue_train.py end-to-end using featurebased data.

    Expected behaviour:
    - train.py produces a saved .keras model
    - continue_train.py loads that model
    - continued training completes successfully
    - a new model file is saved
    """

    # Create tmp directories and write synthetic CSV data
    data_root = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    data_root.mkdir()
    checkpoint_dir.mkdir()

    group_col = "choice_id"
    choice_col = "chosen"

    _write_featurebased_csv(
        data_root / "train.csv",
        group_col=group_col,
        choice_col=choice_col,
        n_groups=6,
        alts_per_group=[3, 4, 3, 4, 3, 4],
    )
    _write_featurebased_csv(
        data_root / "test.csv",
        group_col=group_col,
        choice_col=choice_col,
        n_groups=3,
        alts_per_group=[3, 4, 3],
    )

    # Create tmp config file for train.py
    train_cfg_path = tmp_path / "cfg_train_featurebased.yaml"
    _write_yaml(
        train_cfg_path,
        f"""
data_root: "{data_root}"
checkpoint_dir: "{checkpoint_dir}"

files:
  train_raw: "train.csv"
  test_raw: "test.csv"
  group_col: "{group_col}"
  choice_col: "{choice_col}"

model:
  type: "featurebased"
  depth: 2
  width: 8
  heads: 2

train:
  loss: "nll"
  optimizer: "adam"
  lr: 0.001
  batch_size: 2
  epochs: 1
  patience: null
  l2: 0.0

eval:
  batch_size: 32
  loss: "nll"
""",
    )

    proc_train = _run_train(train_cfg_path)
    assert (
        proc_train.returncode == 0
    ), f"train.py failed.\nstdout:\n{proc_train.stdout}\n\nstderr:\n{proc_train.stderr}"

    base_model_path = _latest_keras(checkpoint_dir)

    # Create tmp config file for continue_train.py
    cont_cfg_path = tmp_path / "cfg_continue_featurebased.yaml"
    _write_yaml(
        cont_cfg_path,
        f"""
data_root: "{data_root}"
checkpoint_dir: "{checkpoint_dir}"

files:
  train_raw: "train.csv"
  test_raw: "test.csv"
  group_col: "{group_col}"
  choice_col: "{choice_col}"

model:
  type: "featurebased"
  load_path: "{base_model_path}"
  save_name: "continued_featurebased"

train:
  loss: "nll"
  optimizer: "adam"
  lr: 0.001
  batch_size: 2
  epochs: 1
  patience: null
  l2: 0.0

eval:
  batch_size: 32
  loss: "nll"
""",
    )

    proc_cont = _run_continue_train(cont_cfg_path)
    assert proc_cont.returncode == 0, (
        "continue_train.py failed.\n"
        f"stdout:\n{proc_cont.stdout}\n\nstderr:\n{proc_cont.stderr}"
    )

    # Load check: confirm the continue run reported a model load
    combined = (proc_cont.stdout + "\n" + proc_cont.stderr).lower()
    assert "load" in combined or "loaded" in combined, (
        "Expected continue_train.py to report loading a model.\n"
        f"stdout:\n{proc_cont.stdout}\n\nstderr:\n{proc_cont.stderr}"
    )

    saved_after = list(checkpoint_dir.glob("*.keras"))
    assert len(saved_after) >= 2, (
        "Expected at least two saved models (base + continued).\n"
        f"stdout:\n{proc_cont.stdout}\n\nstderr:\n{proc_cont.stderr}"
    )


def test_continue_train_fails_missing_load_path(tmp_path: Path) -> None:
    """
    Check continue_train.py fails fast when model.load_path does not exist.
    """

    # Create tmp directories and write synthetic CSV data
    data_root = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    data_root.mkdir()
    checkpoint_dir.mkdir()

    _write_featureless_csv(data_root / "train.csv", n_rows=6, num_items=4)
    _write_featureless_csv(data_root / "test.csv", n_rows=3, num_items=4)

    cfg_path = tmp_path / "cfg_missing_load.yaml"
    _write_yaml(
        cfg_path,
        f"""
data_root: "{data_root}"
checkpoint_dir: "{checkpoint_dir}"

files:
  train_raw: "train.csv"
  test_raw: "test.csv"

model:
  type: "featureless"
  load_path: "{tmp_path / 'does_not_exist.keras'}"

train:
  loss: "nll"
  optimizer: "adam"
  lr: 0.001
  batch_size: 4
  epochs: 1
  patience: null
  l2: 0.0

eval:
  batch_size: 32
  loss: "nll"
""",
    )

    proc = _run_continue_train(cfg_path)
    assert proc.returncode != 0
    combined = (proc.stdout + "\n" + proc.stderr).lower()
    assert (
        "load_path" in combined
        or "does not exist" in combined
        or "not found" in combined
    )
