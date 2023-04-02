from __future__ import annotations

from abc import ABC
from abc import abstractproperty
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np
from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from smac import MultiFidelityFacade
from smac import Scenario
from smac.intensifier import Hyperband

from autoibc.util import TQDMCallback
from autoibc.util import hide_fit_warnings

MAX_RUNTIME = 60 * 59  # Maximum of 1 hour per dataset (-1 minute for cleanup)


class BaseAutoIBC(ABC):
    """Base class for all AutoIBC models.

    All AutoIBC models should inherit from this class and implement the
    `configspace` property.

    Attributes:
        model (Any): The sklearn base model to optimize.
    """

    metric: str = "balanced_accuracy"
    best_config: Configuration | None = None

    def __init__(self, model: Any):
        self.model = model

    @abstractproperty
    def configspace(self) -> ConfigurationSpace:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_model(self, config: dict[str, Any]) -> Any:
        """Returns the sklearn model instantiated with the given configuration.

        This can be overwritten by subclasses to implement custom logic.

        Args:
            config (dict[str, Any]): Configuration of the model

        Returns:
            Any: The sklearn model
        """
        return self.model(**config)

    @staticmethod
    def get_config_dict(config: Configuration) -> dict[str, Any]:
        """Converts a ConfigSpace configuration to a dictionary.

        Handles type conversions, as this is currently not supported
        by ConfigSpace, see (https://github.com/automl/ConfigSpace/issues/95)

        Args:
            config (Configuration): The configuration to convert.

        Returns:
            dict[str, Any]: The configuration as a dictionary.
        """
        config_dict = config.get_dictionary()

        for key, value in config_dict.items():
            if value == "True":
                config_dict[key] = True
            elif value == "False":
                config_dict[key] = False
            elif value == "None":
                config_dict[key] = None
        return config_dict

    @property
    def best_model(self) -> Any:
        """Returns the best model found by the hyperparameter optimization."""
        if not self.best_config:
            raise ValueError("Model was not fitted yet.")
        config_dict = self.get_config_dict(self.best_config)
        return self.get_model(config_dict)

    def target_function(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: int = 5,
    ) -> Callable[[Configuration, int], float]:
        """Returns the target function for SMAC hyperparameter optimization.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
            cv_splits (int, optional): Number of cross-validation splits. Defaults to 5.

        Returns:
            Callable: The target function to optimize.
        """

        @hide_fit_warnings
        def train(cfg: Configuration, budget: float = 1.0, seed: int = 0) -> float:
            config_dict = self.get_config_dict(cfg)
            model = self.get_model(config_dict)
            cv = StratifiedShuffleSplit(
                n_splits=cv_splits,
                train_size=budget,
                random_state=seed,
            )
            scores = cross_val_score(model, X, y, scoring=self.metric, cv=cv)
            return -np.mean(scores)

        return train

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100,
        cv_splits: int = 5,
        min_train_size: float = 0.1,
        run_name: str = "autoibc-run",
        max_runtime: int = MAX_RUNTIME,
        output_dir: Path = Path("results"),
        seed: int = 42,
    ) -> Configuration:
        """Optimizes the hyperparameters of the model with SMAC.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
            n_trials (int, optional): Number of trials to run. Defaults to 100.
            cv_splits (int, optional): Number of cross-validation splits. Defaults to 5.
            min_train_size (float, optional): Minimum training size, used as the budget.
                Defaults to 0.1.
            run_name (str, optional): Name of the run. Defaults to "autoibc-run".
            max_runtime (int, optional): Maximum runtime in seconds.
                Defaults to MAX_RUNTIME.
            output_dir (Path, optional): Output directory for SMAC results. Defaults to
                Path("results").
            seed (int, optional): Random seed. Defaults to 42.

        Returns:
            Configuration: The best configuration found.
        """
        scenario = Scenario(
            configspace=self.configspace,
            name=run_name,
            output_directory=output_dir,
            deterministic=True,
            objectives=[self.metric],
            walltime_limit=max_runtime,
            n_trials=n_trials,
            min_budget=min_train_size,
            max_budget=0.9,  # Keep at least 0.1 for testing
            seed=seed,
        )
        intensifier = Hyperband(
            scenario=scenario,
            eta=2,
            incumbent_selection="highest_observed_budget",
            seed=seed,
        )
        smac = MultiFidelityFacade(
            scenario=scenario,
            target_function=self.target_function(X, y, cv_splits=cv_splits),
            intensifier=intensifier,
            overwrite=True,
            callbacks=[TQDMCallback(metric=self.metric, n_trials=n_trials)],
        )
        self.best_config = smac.optimize()
        self.runtime = smac.intensifier.used_walltime
        return self.best_config

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the labels for the given features."""
        return self.best_model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, cv_splits: int = 10) -> float:
        """Evaluates the model on the given data."""
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        return np.mean(
            cross_val_score(self.best_model, X, y, scoring=self.metric, cv=cv),
        )
