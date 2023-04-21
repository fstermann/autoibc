from __future__ import annotations

import warnings
from functools import wraps
from typing import Any

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Hyperparameter
from sklearn.exceptions import ConvergenceWarning


def hide_fit_warnings(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            # Hide convergence warnings
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return f(*args, **kwargs)

    return wrapper


def convert_special_values(value: Any) -> Any:
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return "None"
    return value


def make_configspace(*hps: Hyperparameter, name: str) -> ConfigurationSpace:
    """Utility function to create a ConfigSpace from a list of Hyperparameters.

    Args:
        *hps (Hyperparameter): List of Hyperparameters
        name (str): Name of the ConfigurationSpace

    Returns:
        ConfigurationSpace: The ConfigurationSpace
    """
    space = {hp.name: hp for hp in hps}
    return ConfigurationSpace(name=name, space=space)


def convert_seconds_to_str(seconds: float) -> str:
    """Converts seconds to a human-readable time string.

    Args:
        seconds (float): Seconds to convert.

    Returns:
        str: Human-readable time string.
    """
    if not isinstance(seconds, float):
        return seconds
    minutes, seconds = divmod(seconds, 60)  # Convert to minutes and seconds
    milliseconds = int(seconds * 1000)  # Convert remaining seconds to milliseconds
    seconds = int(seconds)
    minutes = int(minutes)
    formatted_time = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"[:9]
    return formatted_time
