import os
import json
import yaml
from pathlib import Path
from typing import Any
import shutil
from box import ConfigBox
from src.cnnClassifier import logger

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns"""
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
        logger.info(f"yaml file: {path_to_yaml} loaded successfully")
        return ConfigBox(content)

def create_directories(path_to_directories: list, verbose=True):
    """create list of directories"""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

def save_json(path: Path, data: dict):
    """save json file"""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")

def load_json(path: Path) -> ConfigBox:
    """load json files data"""
    with open(path) as f:
        data = json.load(f)
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(data)

def save_model(model, path: Path):
    """save tensorflow model"""
    model.save(path)
    logger.info(f"model saved at: {path}")

def load_model(path: Path):
    """load tensorflow model"""
    from tensorflow.keras.models import load_model as keras_load_model
    model = keras_load_model(path)
    logger.info(f"model loaded from: {path}")
    return model

def get_class_names_from_directory(directory: Path) -> list:
    """Get class names from directory structure"""
    class_names = sorted([item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))])
    logger.info(f"class names found: {class_names}")
    return class_names
