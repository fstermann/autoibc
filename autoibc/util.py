from __future__ import annotations

import logging
import warnings
from functools import wraps
from typing import Any

import tqdm
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Hyperparameter
from sklearn.exceptions import ConvergenceWarning
from smac import Callback
from smac.main.smbo import SMBO
from smac.runhistory.runhistory import TrialInfo
from smac.runhistory.runhistory import TrialValue


class TQDMCallback(Callback):
    """Custom TQDM callback to display the best metric during optimization.

    Args:
        metric (str): Metric to display
        n_trials (int): Number of trials to run
    """

    def __init__(self, metric: str, n_trials: int) -> None:
        self.metric = metric
        self.pbar = tqdm.tqdm(total=n_trials, desc=f"Best {self.metric}: {{desc}}")

    def on_start(self, smbo: SMBO) -> None:
        pass

    def on_end(self, smbo: SMBO) -> None:
        logging.getLogger("smac").setLevel(logging.INFO)
        self.pbar.close()

    def on_tell_start(
        self,
        smbo: SMBO,
        info: TrialInfo,
        value: TrialValue,
    ) -> bool | None:
        logging.getLogger("smac").setLevel(logging.WARNING)
        return None

    def on_tell_end(
        self,
        smbo: SMBO,
        info: TrialInfo,
        value: TrialValue,
    ) -> bool | None:
        self.pbar.update(1)

        incumbent = smbo.intensifier.get_incumbent()
        if not incumbent:
            return None
        cost = -smbo.runhistory.get_cost(incumbent)
        label = f"Best {self.metric}: {cost:.4f}"
        self.pbar.set_description_str(label)
        return None


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
