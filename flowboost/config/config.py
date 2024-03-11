import importlib.resources as pkg_resources
import logging
import shutil
from pathlib import Path

import tomlkit

DEFAULT_CONFIG_NAME: str = "flowboost_config.toml"


def validate(config: dict) -> bool:
    """Validate the TOML configuration file.

    Args:
        config_path (Path): The path to the TOML configuration file.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    # Check for offload_acquisition
    offload_acquisition = config.get("scheduler", {}).get("offload_acquisition", False)

    if offload_acquisition:
        # Ensure acquisition configuration is present and valid
        if "scheduler.acquisition" not in config:
            logging.error(
                "Offload acquisition is enabled, but `[scheduler.acquisition]` not configured"
            )
            return False

    return True


def create(dir: Path, filename: str = DEFAULT_CONFIG_NAME) -> dict:
    with pkg_resources.path("flowboost.config", DEFAULT_CONFIG_NAME) as config_path:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration template not found [{config_path}]")

        dest = Path(dir, filename)
        if dest.exists():
            raise FileExistsError("Configuration already exists, but create() called")

        shutil.copy(config_path, dest)
        with open(dest, "r") as config_file:
            config = tomlkit.load(config_file)

        if not validate(config):
            raise ValueError("Default configuration is invalid")

        return config


def load(dir: Path, filename: str = DEFAULT_CONFIG_NAME) -> dict:
    fpath = Path(dir, filename)

    if not fpath.exists():
        return create(dir=dir, filename=filename)

    with open(fpath, "r") as toml_f:
        config = tomlkit.load(toml_f)

    if not validate(config):
        raise ValueError("Configuration is invalid")

    return config


def save(config: dict, dir: Path, filename: str = DEFAULT_CONFIG_NAME):
    fpath = Path(dir, filename)

    with open(fpath, "w") as toml_f:
        tomlkit.dump(config, toml_f)
