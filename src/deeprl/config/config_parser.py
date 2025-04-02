# src/deeprl/config/config_parser.py
import os
from copy import deepcopy
from typing import Any, Dict

import yaml


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration.

    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        "env": {
            "id": "CarRacing-v3",
            "render_mode": "human",
            "continuous": False,
        },
        "model": {
            "input_shape": (3, 96, 96),
            "hidden_size": 512,
        },
        "training": {
            "iterations": 30,
            "steps_per_iteration": 2048,
            "initial_human_episodes": 0,
            "intervention_episodes": 3,
            "intervention_threshold": 5,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "batch_size": 64,
            "epochs": 10,
            "random_seed": 42,
        },
        "paths": {
            "model_dir": "models",
            "log_dir": "logs",
        },
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and merge with default config.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        Dict[str, Any]: Merged configuration dictionary
    """
    # Get default configuration
    config = get_default_config()

    # Load user configuration if file exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
            if user_config and isinstance(user_config, dict):
                # Merge configs using a simple nested approach to avoid recursive function calls
                for section_key, section_value in user_config.items():
                    if section_key in config and isinstance(config[section_key], dict) and isinstance(section_value, dict):
                        # This is a nested section, merge it
                        for key, value in section_value.items():
                            config[section_key][key] = value
                    else:
                        # This is a top-level value, just replace it
                        config[section_key] = section_value

    return config


def update_config(base_config: Dict[str, Any], update_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a configuration dictionary with values from another.
    This is a simplified version that only handles two levels of nesting.

    Args:
        base_config (Dict[str, Any]): Base configuration
        update_config (Dict[str, Any]): Configuration to update with

    Returns:
        Dict[str, Any]: Updated configuration
    """
    result = deepcopy(base_config)

    # Update with values from update_config
    for section_key, section_value in update_config.items():
        if section_key in result and isinstance(result[section_key], dict) and isinstance(section_value, dict):
            # This is a nested section, merge it
            for key, value in section_value.items():
                result[section_key][key] = value
        else:
            # This is a top-level value, just replace it
            result[section_key] = section_value

    return result


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config (Dict[str, Any]): Configuration to save
        config_path (str): Path to save the configuration file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)

    # Save config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
