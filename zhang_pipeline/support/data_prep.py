import os
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional


def get_paths(cfg: dict) -> Tuple[str, Optional[str], str, Optional[str]]:
    """
    Extract paths of train/test CSV files from the config dict for both raw and freq data.

    cfg: parsed config dict
    returns: (train_raw_path, train_freq_path, test_raw_path, test_freq_path)
    """
    root = cfg["data_root"]
    files = cfg["files"]

    train_raw_path = os.path.join(root, files["train_raw"])
    test_raw_path = os.path.join(root, files["test_raw"])

    # freq paths are optional
    train_freq_path = None
    test_freq_path = None

    if "train_freq" in files:
        train_freq = files["train_freq"]
        if isinstance(train_freq, str) and train_freq.strip():
            train_freq_path = os.path.join(root, train_freq.strip())
        else:
            train_freq_path = None

    if "test_freq" in files:
        test_freq = files["test_freq"]
        if isinstance(test_freq, str) and test_freq.strip():
            test_freq_path = os.path.join(root, test_freq.strip())
        else:
            test_freq_path = None

    return train_raw_path, train_freq_path, test_raw_path, test_freq_path


def trailing_int(c):
    """
    Given a column name, return the digit at the end of the name.
    """
    i = len(c) - 1
    while i >= 0 and c[i].isdigit():
        i -= 1
    return int(c[i + 1 :])


def extract_featureless_xy(df):
    """
    Extract X and Y columns (X1..XJ, Y1..YJ).

    df: input pandas dataframe from .csv file
    returns: X, Y as numpy arrays
    """

    x_cols = [c for c in df.columns if (c[0] in ("X", "x")) and c[-1].isdigit()]
    x_cols = sorted(x_cols, key=trailing_int)

    y_cols = [c for c in df.columns if (c[0] in ("Y", "y")) and c[-1].isdigit()]
    y_cols = sorted(y_cols, key=trailing_int)

    # convert to numpy
    X = df[x_cols].to_numpy(dtype=float)
    Y = df[y_cols].to_numpy(dtype=float)

    return X, Y


def extract_freq(df):
    """
    Extract Y columns from frequency CSV (column names are numeric).

    df: input DataFrame
    returns: Y as numpy array
    """

    # Columns named "0", "1", ..., "19"
    y_cols = sorted([c for c in df.columns if str(c).isdigit()], key=lambda s: int(s))

    Y = df[y_cols].to_numpy(dtype=float)

    return Y


def extract_featurebased_xy(
    df: pd.DataFrame,
    group_col: str,
    choice_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (X, Y) from a long-format feature-based choice DataFrame.

    Groupby group_col to transform 2d storage into a 3d dataset
    Returns padded arrays:
      X: (N, J, d_x)
      Y: (N, J)
    """

    # All columns except group and choice are features
    exclude = {group_col, choice_col}
    candidate_cols = [c for c in df.columns if c not in exclude]

    # Keep only numeric or boolean columns
    numeric_cols = []
    for c in candidate_cols:
        dtype = df[c].dtype
        if np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_):
            numeric_cols.append(c)

    feature_cols = numeric_cols

    # If any feature col contains NaN, fill with 0
    if df[feature_cols].isna().any().any():
        df[feature_cols] = df[feature_cols].fillna(0.0)

    # group into 3d dataset and drop groups with no chosen alternative
    grouped = df.groupby(group_col)

    groups = []
    for _, sub in grouped:
        # sum of choice_col within this group
        s = float(sub[choice_col].sum())
        # keep only groups with at least one chosen alternative
        # (if you want strictly exactly one, change to: if s == 1.0)
        # group must have at least one offered alternative in X
        feats = sub[feature_cols].to_numpy(dtype=float)
        has_alt = np.any(feats != 0.0)

        if s > 0.0 and has_alt:
            groups.append(sub.reset_index(drop=True))

    if len(groups) == 0:
        raise ValueError(
            f"No choice sets with a chosen alternative found for group_col {group_col!r}"
        )

    # Maximum number of alternatives across valid groups
    max_items = max(len(sub) for sub in groups)

    N = len(groups)
    d_x = len(feature_cols)
    J = max_items

    X = np.zeros((N, J, d_x), dtype=np.float32)
    Y = np.zeros((N, J), dtype=np.float32)

    for i, sub in enumerate(groups):
        k = len(sub)
        # fill first k positions - remainder stays zero (padding)
        X[i, :k, :] = sub[feature_cols].to_numpy(dtype=float)
        Y[i, :k] = sub[choice_col].to_numpy(dtype=float)

    return X, Y


def validate_raw(X, Y):
    """
    Validate raw one-hot X/Y rows.
    returns: None (raises on error)
    """

    # dim checks
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"X and Y must be 2D, got shapes {X.shape} and {Y.shape}.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have same number of rows, got {X.shape[0]} and {Y.shape[0]}."
        )
    if X.shape[1] == 0 or Y.shape[1] == 0:
        raise ValueError(
            f"X and Y must have at least one column; got {X.shape[1]} and {Y.shape[1]}."
        )
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have same number of columns, got {X.shape[1]} and {Y.shape[1]}."
        )

    # X must be 0/1
    if not ((X == 0) | (X == 1)).all():
        raise ValueError("X must contain only 0/1 values.")

    # each row must offer at least one item
    offered = X.sum(axis=1)
    if (offered == 0).any():
        raise ValueError("Found a row in X with no offered items (all zeros).")

    # Y must be non-negative
    if (Y < 0).any():
        raise ValueError("Y contains negative values.")

    # Y rows must sum to 1 (within tolerance)
    row_sums = Y.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Each Y row must sum to 1 for raw one-hot data.")

    # one-hot check: exactly one entry is 1
    # allow small numerical tolerance
    one_hot_counts = (Y > 0.5).sum(axis=1)
    if not np.all(one_hot_counts == 1):
        raise ValueError("Raw Y rows must be one-hot encoded.")

    # enforce that unavailable items have zero probability
    if not np.all(Y[X == 0] == 0):
        raise ValueError("Y assigns probability to unavailable items (X == 0).")


def validate_freq(X, Y):
    """
    Validate frequency-distribution X/Y rows.
    returns: None (raises on error)
    """

    # dim checks
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"X and Y must be 2D, got shapes {X.shape} and {Y.shape}.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have same number of rows, got {X.shape[0]} and {Y.shape[0]}."
        )
    if X.shape[1] == 0 or Y.shape[1] == 0:
        raise ValueError(
            f"X and Y must have at least one column; got {X.shape[1]} and {Y.shape[1]}."
        )
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have same number of columns, got {X.shape[1]} and {Y.shape[1]}."
        )

    # X must be 0/1
    if not ((X == 0) | (X == 1)).all():
        raise ValueError("X must contain only 0/1 values.")

    # each row must offer at least one item
    offered = X.sum(axis=1)
    if (offered == 0).any():
        raise ValueError("Found a row in X with no offered items (all zeros).")

    # Y must be non-negative
    if (Y < 0).any():
        raise ValueError("Y contains negative values.")

    # Y rows must sum to 1 (within tolerance)
    row_sums = Y.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Each Y row must sum to 1 for frequency data.")

    # enforce that unavailable items have zero probability
    if not np.all(Y[X == 0] == 0):
        raise ValueError("Y assigns probability to unavailable items (X == 0).")


def validate_featurebased(X: np.ndarray, Y: np.ndarray) -> None:
    """
    Validate padded feature-based (X, Y) arrays.

    X: shape (N, J, d_x)
    Y: shape (N, J)

    Checks:
      - Correct ranks and matching leading dimensions.
      - Y is non-negative.
      - Each Y row sums to 1 (within tolerance).
      - Each row has at least one offered item (sum(Y[i, :]) > 0).
    """

    if X.ndim != 3:
        raise ValueError(f"Feature-based X must be 3D (N, J, d_x), got ndim={X.ndim}.")
    if Y.ndim != 2:
        raise ValueError(f"Feature-based Y must be 2D (N, J), got ndim={Y.ndim}.")

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Mismatched shapes: X{X.shape} and Y{Y.shape} must have same N and J."
        )
    if X.shape[2] == 0:
        raise ValueError(
            "Feature-based X must have at least one feature dimension (d_x > 0)."
        )

    # Y must be non-negative
    if (Y < 0).any():
        raise ValueError("Y contains negative values.")

    # Each row must have at least one chosen alternative
    row_sums = Y.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("Found a row in Y with no chosen alternative (sum <= 0).")

    # Y rows must sum to 1 (single choice per set, including padding zeros)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Each Y row must sum to 1 for feature-based data.")


def to_dataset(X, Y, batch_size: int):
    """
    Convert NumPy arrays to tensors and build a tf.data.Dataset.

    X, Y: NumPy arrays.
    returns: (dataset, X_tensor, Y_tensor)
    """

    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    Y_tensor = tf.convert_to_tensor(Y, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((X_tensor, Y_tensor))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, X_tensor, Y_tensor


def load_csv(path: str):
    """
    Load csv file into a DataFrame.

    path: file path.
    returns: pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    if df.shape[0] == 0:
        raise ValueError(f"CSV file is empty: {path}")
    if df.shape[1] == 0:
        raise ValueError(f"CSV file has no columns: {path}")
    if df.isna().all().all():
        raise ValueError(f"CSV file contains only NaN values: {path}")

    return df


def load_featureless_file(path: str):
    """
    Load a single CSV file and return X, Y arrays.

    path: csv file path.
    returns: (X, Y) as numpy arrays.
    """

    df = load_csv(path)
    X, Y = extract_featureless_xy(df)
    validate_raw(X, Y)

    return X, Y


def load_freq_file(path: str):
    """
    Load a frequency CSV file and return Y only.

    path: csv file path.
    returns: Y as numpy array.
    """

    df = load_csv(path)
    Y = extract_freq(df)

    return Y


def load_featurebased_file(
    path: str,
    group_col: str,
    choice_col: str,
):
    """
    Load .csv file and return validated (X, Y) arrays.

    path: csv file path.
    group_col: column grouping rows into choice sets (used to store 3d data in 2d shape).
    choice_col: 0/1 indicator of chosen alternative within each set.

    Returns:
      X: (N, J, d_x)
      Y: (N, J)
    """

    df = load_csv(path)

    # check required columns exist
    if group_col not in df.columns:
        raise ValueError(f"Missing group_col '{group_col}' in file: {path}")
    if df[group_col].nunique() == 0:
        raise ValueError(
            f"No groups found using group_col '{group_col}' in file: {path}"
        )
    if choice_col not in df.columns:
        raise ValueError(f"Missing choice_col '{choice_col}' in file: {path}")

    X, Y = extract_featurebased_xy(df, group_col, choice_col)
    validate_featurebased(X, Y)

    return X, Y


def load_featureless_data(cfg: dict, batch_size: int):
    """
    Load full train/test featureless data and build datasets.

    returns:
    train_ds - dataset of train data
    X_train
    Y_train
    test_ds - dataset of test data
    X_test
    Y_test
    freq_data_dict - dict with frequency based labels for train and test data
        freq based data averages Y labels over the dataset to smooth predictions
    """

    # extract paths from config
    train_raw_path, train_freq_path, test_raw_path, test_freq_path = get_paths(cfg)

    # load raw data (always required)
    X_train_raw, Y_train_raw = load_featureless_file(train_raw_path)
    X_test_raw, Y_test_raw = load_featureless_file(test_raw_path)

    # build train/test datasets from raw tensors
    train_ds, X_train, Y_train = to_dataset(X_train_raw, Y_train_raw, batch_size)
    test_ds, X_test, Y_test = to_dataset(X_test_raw, Y_test_raw, batch_size)

    # load freq data if available
    freq_data = {}

    if train_freq_path is not None:
        Y_train_f = load_freq_file(train_freq_path)
        # check num of labels matches
        if Y_train_f.shape != Y_train_raw.shape:
            raise ValueError(
                f"Train frequency data shape {Y_train_f.shape} "
                f"does not match raw labels shape {Y_train_raw.shape}."
            )
        validate_freq(X_train_raw, Y_train_f)
        freq_data["train_freq"] = Y_train_f

    if test_freq_path is not None:
        Y_test_f = load_freq_file(test_freq_path)
        # check num of labels matches
        if Y_test_f.shape != Y_test_raw.shape:
            raise ValueError(
                f"Test frequency data shape {Y_test_f.shape} "
                f"does not match raw labels shape {Y_test_raw.shape}."
            )
        validate_freq(X_test_raw, Y_test_f)
        freq_data["test_freq"] = Y_test_f

    return (train_ds, X_train, Y_train, test_ds, X_test, Y_test, freq_data)


def load_featurebased_data(cfg: dict, batch_size: int):
    """
    Load full train/test featurebased data and build datasets.

    returns:
    train_ds - dataset of train data
    X_train
    Y_train
    test_ds - dataset of test data
    X_test
    Y_test
    freq_data_dict - empty so far
    """

    # 1) Extract paths from config
    train_raw_path, train_freq_path, test_raw_path, test_freq_path = get_paths(cfg)

    # 2) Extract grouping & choice columns from config
    group_col = cfg["files"]["group_col"]
    choice_col = cfg["files"]["choice_col"]

    # 3) Load raw (X, Y) arrays
    X_train_raw, Y_train_raw = load_featurebased_file(
        train_raw_path, group_col, choice_col
    )
    X_test_raw, Y_test_raw = load_featurebased_file(
        test_raw_path, group_col, choice_col
    )

    # 5) Convert to tf.data.Dataset
    train_ds, X_train, Y_train = to_dataset(X_train_raw, Y_train_raw, batch_size)
    test_ds, X_test, Y_test = to_dataset(X_test_raw, Y_test_raw, batch_size)

    # 6) No freq-based data for feature-based models
    freq_data = {}

    return (train_ds, X_train, Y_train, test_ds, X_test, Y_test, freq_data)
