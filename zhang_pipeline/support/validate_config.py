import os
from typing import Any, Dict

# Global defaults and allowed sets

_DEFAULT_TRAIN = {
    "loss": "nll",
    "optimizer": "adam",
    "lr": 0.0005,
    "batch_size": 1024,
    "epochs": 5,
    "patience": None,  # None => no early stopping
    "l2": 0.001,
}

_ALLOWED_LOSSES = {"mse", "nll"}  # always lowercase
_ALLOWED_OPTIMIZERS = {"adam", "sgd"}  # always lowercase
_ALLOWED_MODEL_TYPES = {"featureless", "featurebased", "stacked"}  # always lowercase

_DEFAULT_EVAL_BATCH_SIZE = 4096
_DEFAULT_ARGS_SEED = 42


# Small helpers - convert arg to specified type if possible


def _as_mapping(cfg: Any, name: str) -> Dict[str, Any]:
    """
    Ensure cfg is a dict.
    None -> empty dict.
    """
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return dict(cfg)


def _as_str(value: Any, field: str) -> str:
    """
    Converts value to string.
    """
    try:
        return str(value)
    except Exception:
        raise ValueError(f"{field} must be convertible to string, got {value}.")


def _as_int(value: Any, field: str) -> int:
    """
    Convert to int.
    """
    try:
        return int(value)
    except Exception:
        raise ValueError(f"{field} must be convertible to int, got {value}.")


def _as_float(value: Any, field: str) -> float:
    """
    Convert to float.
    """
    try:
        return float(value)
    except Exception:
        raise ValueError(f"{field} must be convertible to float, got {value}.")


# Section normalisers for each part of the config file


def _normalise_load(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise the model section config file for the calls that load a model.

    model is a mapping
    model.type is a supported type
    model.load_path exists on disk
    """
    model = _as_mapping(cfg.get("model"), "model")

    # type
    if "type" not in model:
        raise ValueError("model.type must be provided for evaluate/continue configs.")
    model_type = _as_str(model.get("type", "featureless"), "model.type").lower()
    if model_type not in _ALLOWED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model type {model_type!r}. "
            f"Allowed model types are {_ALLOWED_MODEL_TYPES}."
        )
    model["type"] = model_type

    # load_path
    load_path = _as_str(model.get("load_path", ""), "model.load_path").strip()
    if not load_path:
        raise ValueError(
            "model.load_path must be provided for evaluate/continue configs."
        )
    if not os.path.isfile(load_path):
        raise ValueError(
            f"model.load_path does not exist or is not a file: {load_path!r}"
        )

    model["load_path"] = load_path

    # save_name: optional, but if present must be non-empty string
    if "save_name" in model:
        model["save_name"] = _as_str(model["save_name"], "model.save_name").strip()
    else:
        model["save_name"] = ""

    cfg["model"] = model
    return cfg


def _normalise_model_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the 'model' section for the main train code.

    check:
    model.type is allowed
    model.depth is between 1 and 16 (default 5)
    model.width is between 2 and 1024 (default 300)
    model.heads is between 1 and 32 for the featurebased model
    model.save_name is valid
    """
    model = _as_mapping(cfg.get("model"), "model")

    # model.type
    if "type" not in model:
        raise ValueError("model.type must be provided for train configs.")
    model_type = _as_str(model.get("type"), "model.type").lower()

    if model_type not in _ALLOWED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model type {model_type!r};"
            f"Allowed model types are {_ALLOWED_MODEL_TYPES}."
        )
    model["type"] = model_type

    # depth must be between 1 and 16 - default is 5
    depth = _as_int(model.get("depth", 5), "model.depth")
    if not 1 <= depth <= 16:
        raise ValueError("model.depth must be between 1 and 16.")
    model["depth"] = depth

    # width: must be between 2 and 1024 - default is 300
    width = _as_int(model.get("width", 300), "model.width")
    if not 2 <= width <= 1024:
        raise ValueError("model.width must be between 2 and 1024.")
    model["width"] = width

    # heads: must be between 1 and 32 - default is 8
    # only meaningful for featurebased models
    if model_type == "featurebased":
        heads = _as_int(model.get("heads", 8), "model.heads")
        if not 1 <= heads <= 32:
            raise ValueError(
                "model.heads must be between 1 and 32 for featurebased models."
            )
        model["heads"] = heads

    # save_name: optional, but if present must be non-empty string
    if "save_name" in model:
        model["save_name"] = _as_str(model["save_name"], "model.save_name").strip()
    else:
        model["save_name"] = ""

    cfg["model"] = model
    return cfg


def _normalise_files_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the 'files' section of the config.

    For featureless models:
    train_raw, test_raw must be present.

    For featurebased models:
    train_raw, test_raw must be present.
    group_col: column grouping rows into choice sets.
    choice_col: bool column indicating the chosen item.
    """
    files = _as_mapping(cfg.get("files"), "files")

    # data_root must exist and be a directory
    data_root = _as_str(cfg.get("data_root", ""), "data_root").strip()
    if not data_root:
        raise ValueError("data_root must be provided in the config.")
    if not os.path.isdir(data_root):
        raise ValueError(f"data_root directory does not exist: {data_root!r}")

    # Required CSVs
    train_raw = _as_str(files.get("train_raw", ""), "files.train_raw").strip()
    test_raw = _as_str(files.get("test_raw", ""), "files.test_raw").strip()
    if not train_raw:
        raise ValueError("files.train_raw must be provided.")
    if not test_raw:
        raise ValueError("files.test_raw must be provided.")

    # Full paths
    full_train_raw = os.path.join(data_root, train_raw)
    full_test_raw = os.path.join(data_root, test_raw)

    # Existence checks
    if not os.path.isfile(full_train_raw):
        raise FileNotFoundError(f"files.train_raw does not exist: {full_train_raw}")
    if not os.path.isfile(full_test_raw):
        raise FileNotFoundError(f"files.test_raw does not exist: {full_test_raw}")

    files["train_raw"] = train_raw
    files["test_raw"] = test_raw

    # Only when model is featurebased
    model_type = cfg["model"]["type"]
    if model_type == "featurebased":
        group_col = files.get("group_col", None)
        choice_col = files.get("choice_col", None)

        if group_col is None:
            raise ValueError("Featurebased model requires files.group_col.")
        if choice_col is None:
            raise ValueError("Featurebased model requires files.choice_col.")

        group_col = _as_str(group_col, "files.group_col").strip()
        choice_col = _as_str(choice_col, "files.choice_col").strip()

        if not group_col:
            raise ValueError("files.group_col must be a non-empty string.")
        if not choice_col:
            raise ValueError("files.choice_col must be a non-empty string.")

        files["group_col"] = group_col
        files["choice_col"] = choice_col

    cfg["files"] = files
    return cfg


def _normalise_train_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the train section of the config.

    check:
    train.loss is supported
    train.optimizer is supported
    train.lr is between 1e-5 and 1e-2 (default 0.001).
    train.batch_size is between 1 and 8192 (default 1024).
    train.epochs is between 1 and 1000 (default 10).
    train.patience is None or between 1 and min(epochs, 1000).
    train.l2 is between 0 and 0.001
    """

    train = _as_mapping(cfg.get("train"), "train")

    # loss
    loss_fn = train.get("loss", _DEFAULT_TRAIN["loss"])
    loss_fn = _as_str(loss_fn, "train.loss").lower()
    if loss_fn not in _ALLOWED_LOSSES:
        raise ValueError(
            f"train.loss {loss_fn} is not supported; "
            f"allowed values are {_ALLOWED_LOSSES}."
        )
    train["loss"] = loss_fn

    # optimizer
    opt = train.get("optimizer", _DEFAULT_TRAIN["optimizer"])
    opt = _as_str(opt, "train.optimizer").lower()
    if opt not in _ALLOWED_OPTIMIZERS:
        raise ValueError(
            f"train.optimizer {opt} is not supported; "
            f"allowed values are {_ALLOWED_OPTIMIZERS}."
        )
    train["optimizer"] = opt

    # learning rate betweem 1e-5 and 1e-2
    lr = _as_float(train.get("lr", _DEFAULT_TRAIN["lr"]), "train.lr")
    if not 0.00001 <= lr <= 0.01:
        raise ValueError("train.lr must be between 1e-5 and 1e-2")
    train["lr"] = lr

    # batch size between 1 and 8192
    batch_size = _as_int(
        train.get("batch_size", _DEFAULT_TRAIN["batch_size"]), "train.batch_size"
    )
    if not 1 <= batch_size <= 8192:
        raise ValueError("train.batch_size must be between 1 and 8192")
    train["batch_size"] = batch_size

    # epochs between 1 and 1000
    epochs = _as_int(train.get("epochs", _DEFAULT_TRAIN["epochs"]), "train.epochs")
    if not 1 <= epochs <= 1000:
        raise ValueError("train.epochs must be between 1 and 1000")
    train["epochs"] = epochs

    # patience: None or between 1 and 1000 - 0 treated as None
    patience = train.get("patience", _DEFAULT_TRAIN["patience"])
    if patience is None or patience == 0:
        patience = None
    else:
        patience = _as_int(patience, "train.patience")
        max_patience = min(epochs, 1000)
        if not 1 <= patience <= max_patience:
            raise ValueError(f"train.patience must be between 1 and {max_patience}")
    train["patience"] = patience

    # L2 regularization strength is between 0 and 0.001 - 0 means no regularization
    l2 = train.get("l2", _DEFAULT_TRAIN["l2"])
    l2 = _as_float(l2, "train.l2")
    if not 0.0 <= l2 <= 0.001:
        raise ValueError("train.l2 must be between 0 and 0.001")
    train["l2"] = l2

    cfg["train"] = train
    return cfg


def _normalise_eval_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the 'eval' section of the config.

    check:
    eval.batch_size is between 1 and 16384 (default 4096).
    eval.loss is a supported loss name default 'nll'.
    """

    eval = _as_mapping(cfg.get("eval"), "eval")

    # batch_size: between 1 and 16384
    if "batch_size" in eval:
        batch_size = _as_int(eval["batch_size"], "eval.batch_size")
        if not 1 <= batch_size <= 16384:
            raise ValueError("eval.batch_size must be between 1 and 16384")
        eval["batch_size"] = batch_size
    else:
        eval["batch_size"] = _DEFAULT_EVAL_BATCH_SIZE

    # loss: must be defined in _ALLOWED_LOSSES:
    if "loss" in eval:
        loss_fn = eval["loss"]
        loss_fn = _as_str(loss_fn, "eval.loss").lower()
        if loss_fn not in _ALLOWED_LOSSES:
            raise ValueError(
                f"eval.loss {loss_fn!r} is not supported; "
                f"allowed values are {_ALLOWED_LOSSES}."
            )
        eval["loss"] = loss_fn
    else:
        eval["loss"] = "nll"

    cfg["eval"] = eval
    return cfg


def _normalise_checkpoint_dir(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure cfg['checkpoint_dir'] exists and is a non-empty string.
    Default: 'checkpoints'
    """
    value = cfg.get("checkpoint_dir", "checkpoints")
    value = _as_str(value, "checkpoint_dir").strip()

    if not value:
        raise ValueError("checkpoint_dir must be a non-empty string.")

    cfg["checkpoint_dir"] = value
    return cfg


# Public validation functions


def validate_train_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalise the configuration for train.py.
    Mutates cfg in-place and returns it.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)}")

    cfg = _normalise_checkpoint_dir(cfg)
    cfg = _normalise_model_section(cfg)
    cfg = _normalise_files_section(cfg)
    cfg = _normalise_train_section(cfg)
    cfg = _normalise_eval_section(cfg)

    return cfg


def validate_continue_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalise configuration for continue_train.py.
    Mutates cfg in-place and returns it.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)}.")

    cfg = _normalise_checkpoint_dir(cfg)
    cfg = _normalise_load(cfg)
    cfg = _normalise_files_section(cfg)
    cfg = _normalise_train_section(cfg)
    cfg = _normalise_eval_section(cfg)

    return cfg


def validate_eval_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalise configuration for evaluate.py.
    Mutates cfg in-place and returns it.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)}.")

    cfg = _normalise_checkpoint_dir(cfg)
    cfg = _normalise_load(cfg)
    cfg = _normalise_files_section(cfg)
    cfg = _normalise_eval_section(cfg)

    return cfg
