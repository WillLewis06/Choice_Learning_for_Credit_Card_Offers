import os
import yaml
import argparse
from typing import Dict


def load_config() -> dict:
    """
    Load _config.yaml file from the argparse args and return it as a dict.

    returns cfg: dict of parsed config
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = args.config

    if not isinstance(config_path, str) or not config_path:
        raise ValueError(
            f"Invalid config path in args.config: {config_path!r}"
            f"please run as command.py --config <config_name>.yaml"
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 2. Load YAML
    with open(config_path, "r") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {e}")

    if not isinstance(cfg, dict):
        raise ValueError("Config file did not contain a dictionary at top level.")

    # 3. Attach all CLI args as a sub-dict
    cfg["args"] = dict(vars(args))

    return cfg
