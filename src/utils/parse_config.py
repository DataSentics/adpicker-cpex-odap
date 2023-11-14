from omegaconf import OmegaConf
import os
from typing import Dict

 
def parse_config_file(yaml_path, overrides: Dict = None):
    """Function to parse the config file and override the env variable if needed"""
    
    config = OmegaConf.load(yaml_path)
    saved_env_vars = dict(os.environ)
    
    # If there are values that needs to be overrided, update the env variable and after using it return it to the original state
    if overrides:
            os.environ.update(overrides)
            resolved_cnfg = OmegaConf.to_container(config, resolve = True)
            os.environ.update(saved_env_vars)
    else:    
        resolved_cnfg = OmegaConf.to_container(config, resolve = True)    
    return resolved_cnfg

