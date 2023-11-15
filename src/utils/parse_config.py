import os
from typing import Dict
from omegaconf import OmegaConf

from src.utils.find_config_path import build_path


def parse_config_file(yaml_path, overrides: Dict = None):
    """Function to parse the config file and override the env variable if needed"""

    config = OmegaConf.load(yaml_path)
    saved_env_vars = dict(os.environ)

    # If there are values that needs to be overrided,
    # update the env variable and after using it return it to the original state
    if overrides:
        os.environ.update(overrides)
        OmegaConf.resolve(config)
        os.environ.update(saved_env_vars)
    else:
        OmegaConf.resolve(config)
    return config

config_file_path = build_path()
cnfg_file = parse_config_file(config_file_path)
