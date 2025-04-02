import inspect
import json
import sys
from typing import Any, Dict, List

import gymnasium as gym


def get_env_kwargs(env_id: str) -> Dict[str, Any]:
    """
    Get the keyword arguments for a specific gym environment.

    Args:
        env_id: The ID of the environment to inspect

    Returns:
        Dictionary of parameter names and their default values
    """
    try:
        env_spec = gym.spec(env_id)
        env_creator = env_spec.entry_point

        # If env_creator is a string, import it
        if isinstance(env_creator, str):
            module_name, class_name = env_creator.rsplit(":", 1)
            module = __import__(module_name, fromlist=[class_name])
            env_class = getattr(module, class_name)
        else:
            env_class = env_creator

        # Inspect the class's __init__ method to get the parameters
        signature = inspect.signature(env_class.__init__)
        parameters = signature.parameters

        # Create a dictionary of parameter names and default values
        kwargs = {}
        for name, param in parameters.items():
            # Skip 'self' and any parameters without defaults
            if name == "self" or param.default is inspect.Parameter.empty:
                continue
            kwargs[name] = param.default

        return kwargs

    except Exception as e:
        print(f"Error getting parameters for environment {env_id}: {e}", file=sys.stderr)
        return {}


def list_envs() -> List[str]:
    """
    List all available environments in Gym.

    Returns:
        List of environment IDs
    """
    return [env_spec.id for env_spec in gym.envs.registry.values()]


def build_env(env_id: str, **kwargs) -> gym.Env:
    """
    Build a Gym environment with the specified parameters.

    Args:
        env_id: The ID of the environment to build
        **kwargs: Parameters to pass to the environment constructor

    Returns:
        The constructed Gym environment
    """
    return gym.make(env_id, **kwargs)


def parse_value(value_str: str) -> Any:
    """
    Parse a string value into an appropriate Python type.

    Args:
        value_str: The string to parse

    Returns:
        The parsed value
    """
    try:
        # Try to parse as JSON first (for lists, dicts, bools, null)
        return json.loads(value_str)
    except json.JSONDecodeError:
        # If that fails, return as string
        return value_str
