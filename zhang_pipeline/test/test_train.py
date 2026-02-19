"""
End-to-end pytest tests for train.py.

These tests check that:
- train.py can be executed using only a YAML config file
- valid inputs run to completion and save a model
- invalid inputs get caught early

The tests treat train.py as a CLI entrypoint with the config file passed as an option
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


def _run_train(cfg_path: Path) -> subprocess.CompletedProcess[str]:
    """
    Executes train.py as a subprocess using a given config file.
    Enforces CPU for simplicity.
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


def _assert_saved_and_finite(
    proc: subprocess.CompletedProcess[str],
    checkpoint_dir: Path,
) -> None:
    """
    Assert that a training run completed and saved a .keras model file.
    """
    assert (
        proc.returncode == 0
    ), f"train.py failed.\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"

    saved_models = list(checkpoint_dir.glob("*.keras"))
    assert (
        saved_models
    ), f"No model saved.\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"

    match = re.search(
        r"Model best .* on raw data:\s*([0-9eE\.\+\-]+)",
        proc.stdout,
    )
    assert match, f"Could not parse final metric.\nstdout:\n{proc.stdout}"

    metric = float(match.group(1))
    assert math.isfinite(metric), f"Final metric not finite: {metric}"


def _write_featureless_csv(
    path: Path,
    n_rows: int,
    num_items: int,
) -> None:
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


def test_train_featureless_runs(tmp_path: Path) -> None:
    """
    Run train.py end-to-end using featureless data.

    Expected behaviour:
    - training completes successfully
    - a model file is saved
    - a finite evaluation metric is reported
    """

    # Create tmp directories with the required config and .csv files
    data_root = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    data_root.mkdir()
    checkpoint_dir.mkdir()

    _write_featureless_csv(data_root / "train.csv", n_rows=12, num_items=4)
    _write_featureless_csv(data_root / "test.csv", n_rows=6, num_items=4)

    cfg_path = tmp_path / "cfg_featureless.yaml"
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

    proc = _run_train(cfg_path)
    _assert_saved_and_finite(proc, checkpoint_dir)


def test_train_featurebased_runs(tmp_path: Path) -> None:
    """
    Verify that train.py runs end-to-end using featurebased data.

    Expected behaviour:
    - featurebased data is grouped and padded correctly
    - training completes successfully
    - a model file is saved
    """

    # Create tmp directories with the required config and .csv files
    data_root = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    data_root.mkdir()
    checkpoint_dir.mkdir()

    group_col = "choice_id"
    choice_col = "chosen"

    # Create tmp train.csv data
    _write_featurebased_csv(
        data_root / "train.csv",
        group_col=group_col,
        choice_col=choice_col,
        n_groups=6,
        alts_per_group=[3, 4, 3, 4, 3, 4],
    )

    # Create tmp test.csv data
    _write_featurebased_csv(
        data_root / "test.csv",
        group_col=group_col,
        choice_col=choice_col,
        n_groups=3,
        alts_per_group=[3, 4, 3],
    )

    # Create tmp config file
    cfg_path = tmp_path / "cfg_featurebased.yaml"
    _write_yaml(
        cfg_path,
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

    proc = _run_train(cfg_path)
    _assert_saved_and_finite(proc, checkpoint_dir)


def test_train_fails_missing_files(tmp_path: Path) -> None:
    """
    Check train.py fails fast when the .csv files don't exist.
    """

    data_root = tmp_path / "data"
    ckpt_dir = tmp_path / "checkpoints"
    data_root.mkdir()
    ckpt_dir.mkdir()

    cfg_path = tmp_path / "cfg_missing_files.yaml"
    _write_yaml(
        cfg_path,
        f"""
data_root: "{data_root}"
checkpoint_dir: "{ckpt_dir}"

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

    proc = _run_train(cfg_path)
    assert proc.returncode != 0
    combined = (proc.stdout + "\n" + proc.stderr).lower()
    assert "does not exist" in combined or "not found" in combined


def test_train_fails_out_of_bounds_depth(tmp_path: Path) -> None:
    """
    Check train.py fails fast when model.depth is outside the allowed range.
    """

    data_root = tmp_path / "data"
    ckpt_dir = tmp_path / "checkpoints"
    data_root.mkdir()
    ckpt_dir.mkdir()

    j = 4
    _write_featureless_csv(data_root / "train.csv", 6, j)
    _write_featureless_csv(data_root / "test.csv", 3, j)

    cfg_path = tmp_path / "cfg_bad_depth.yaml"
    _write_yaml(
        cfg_path,
        f"""
data_root: "{data_root}"
checkpoint_dir: "{ckpt_dir}"

files:
  train_raw: "train.csv"
  test_raw: "test.csv"

model:
  type: "featureless"
  depth: 999
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

    proc = _run_train(cfg_path)
    assert proc.returncode != 0
    combined = (proc.stdout + "\n" + proc.stderr).lower()
    assert "model.depth" in combined or "between 1 and 16" in combined


def test_train_fails_featureless_invalid_x(tmp_path: Path) -> None:
    """
    Check train.py fails fast when featureless X contains values outside {0,1}.
    """
    data_root = tmp_path / "data"
    ckpt_dir = tmp_path / "checkpoints"
    data_root.mkdir()
    ckpt_dir.mkdir()

    # Make an invalid featureless CSV: X contains a 2 (not binary)
    df = pd.DataFrame(
        [
            {"X1": 1, "X2": 0, "X3": 2, "X4": 0, "Y1": 1, "Y2": 0, "Y3": 0, "Y4": 0},
        ]
    )
    df.to_csv(data_root / "train.csv", index=False)
    df.to_csv(data_root / "test.csv", index=False)

    cfg_path = tmp_path / "cfg_bad_x.yaml"
    _write_yaml(
        cfg_path,
        f"""
data_root: "{data_root}"
checkpoint_dir: "{ckpt_dir}"

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
  batch_size: 1
  epochs: 1

eval:
  batch_size: 32
  loss: "nll"
""",
    )

    proc = _run_train(cfg_path)
    assert proc.returncode != 0
    combined = (proc.stdout + "\n" + proc.stderr).lower()
    assert "0/1" in combined or "x must contain only" in combined


def test_train_fails_featurebased_no_choices(tmp_path: Path) -> None:
    """
    Check train.py fails fast when featurebased data contains no chosen alternative.
    """
    data_root = tmp_path / "data"
    ckpt_dir = tmp_path / "checkpoints"
    data_root.mkdir()
    ckpt_dir.mkdir()

    group_col = "choice_id"
    choice_col = "chosen"

    # All chosen=0 -> loader drops every group -> should raise
    rows = []
    for g in range(3):
        for a in range(3):
            rows.append(
                {group_col: g, "price": 10.0 + a, "quality": 0.7, choice_col: 0}
            )
    pd.DataFrame(rows).to_csv(data_root / "train.csv", index=False)
    pd.DataFrame(rows).to_csv(data_root / "test.csv", index=False)

    cfg_path = tmp_path / "cfg_no_choices.yaml"
    _write_yaml(
        cfg_path,
        f"""
data_root: "{data_root}"
checkpoint_dir: "{ckpt_dir}"

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

eval:
  batch_size: 32
  loss: "nll"
""",
    )

    proc = _run_train(cfg_path)
    assert proc.returncode != 0
    combined = (proc.stdout + "\n" + proc.stderr).lower()
    assert (
        "no choice sets" in combined
        or "chosen alternative" in combined
        or "sum" in combined
    )


def test_train_fails_featurebased_missing_group_choice_cols(tmp_path: Path) -> None:
    """
    Check train.py fails fast when the featurebased config is missing group_col or choice_col.
    """
    data_root = tmp_path / "data"
    ckpt_dir = tmp_path / "checkpoints"
    data_root.mkdir()
    ckpt_dir.mkdir()

    # Create a valid-looking featurebased CSV, but omit group_col/choice_col from config
    rows = [
        {"choice_id": 0, "price": 10.0, "quality": 0.7, "chosen": 1},
        {"choice_id": 0, "price": 12.0, "quality": 0.6, "chosen": 0},
    ]
    pd.DataFrame(rows).to_csv(data_root / "train.csv", index=False)
    pd.DataFrame(rows).to_csv(data_root / "test.csv", index=False)

    cfg_path = tmp_path / "cfg_missing_cols.yaml"
    _write_yaml(
        cfg_path,
        f"""
data_root: "{data_root}"
checkpoint_dir: "{ckpt_dir}"

files:
  train_raw: "train.csv"
  test_raw: "test.csv"
  # group_col and choice_col intentionally missing

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

eval:
  batch_size: 32
  loss: "nll"
""",
    )

    proc = _run_train(cfg_path)
    assert proc.returncode != 0
    combined = (proc.stdout + "\n" + proc.stderr).lower()
    assert "group_col" in combined or "choice_col" in combined
