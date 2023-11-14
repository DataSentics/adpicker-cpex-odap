from pathlib import Path
import os


def build_path():
    """Build the path to the config file
    Start from current file and going back up until the root or 
    until it finds the src directory where the config is located"""
    
    cnfg_path = os.path.dirname(__file__)
    while cnfg_path != '/':
        if 'src' in os.listdir(cnfg_path):
            return Path(cnfg_path) / 'src/config/config.yaml'
        cnfg_path = os.path.dirname(cnfg_path)  
    return cnfg_path

