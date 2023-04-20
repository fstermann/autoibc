from __future__ import annotations

import time
from abc import ABC
from abc import abstractproperty
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np
from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y
from smac import MultiFidelityFacade
from smac import Scenario
from smac.intensifier import Hyperband

from autoibc.util import TQDMCallback
from autoibc.util import hide_fit_warnings

MAX_RUNTIME = 60 * 59  # Maximum of 1 hour per dataset (-1 minute for cleanup)


class BaseAutoIBC(BaseEstimator, ABC):
    """Base class for all AutoIBC models.

    All AutoIBC models should inherit from this class and implement the
    `configspace` property.

    Attributes:
        model (Any): The sklearn base model to optimize.
    """

    estimator: BaseEstimator
    metric: str = "balanced_accuracy"
    best_config: Configuration | None = None

    def __init__(self, estimator: BaseEstimator) -> None:
        super().__init__()
        self.estimator = estimator

    @abstractproperty
    def configspace(self) -> ConfigurationSpace:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def set_params(self, **params: Any) -> BaseAutoIBC:
        """Sets the parameters of the model.

        Args:
            **params (Any): Parameters to set.

        Returns:
            BaseAutoIBC: The model with the updated parameters.
        """
        params = self._prepare_params(**params)
        return super().set_params(**params)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Returns the parameters of the best model."""
        if not deep:
            return {}
        params = self.estimator.get_params(deep=deep)
        params["estimator"] = self.estimator
        return self._prepare_params(**params)

    def _unprepare_params(self, **params: Any) -> dict[str, Any]:
        """Unprepares the parameters of the model.

        This can be overwritten by subclasses to implement custom logic.

        Args:
            **params (Any): Parameters to unprepare

        Returns:
            dict[str, Any]: The unprepared parameters
        """
        return {
            (k if not k.startswith("estimator__") else k.replace("estimator__", "")): v
            for k, v in params.items()
        }

    def _prepare_params(self, **params: Any) -> dict[str, Any]:
        """Prepares the parameters for the model.

        This can be overwritten by subclasses to implement custom logic.

        Args:
            **params (Any): Parameters to prepare

        Returns:
            dict[str, Any]: The prepared parameters
        """
        return {
            (k if k.startswith("estimator") else f"estimator__{k}"): v
            for k, v in params.items()
        }

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
            params = self._prepare_params(**config_dict)
            self.set_params(**params)
            cv = StratifiedShuffleSplit(
                n_splits=cv_splits,
                train_size=budget,
                random_state=seed,
            )
            scores = cross_val_score(self.estimator, X, y, scoring=self.metric, cv=cv)
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
        **fit_params,
    ) -> BaseAutoIBC:
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
        if not fit_params.pop("in_optimization", True):
            return self.estimator.fit(X, y, **fit_params)

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        scenario = Scenario(
            configspace=self.configspace,
            name=f"{run_name}/{time.time()}",
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
            target_function=self.target_function(self.X_, self.y_, cv_splits=cv_splits),
            intensifier=intensifier,
            overwrite=True,
            callbacks=[TQDMCallback(metric=self.metric, n_trials=n_trials)],
        )
        self.best_config = smac.optimize()

        self.runtime = smac.intensifier.used_walltime

        print(self.best_config)
        if self.best_config:
            params = self._prepare_params(**self.get_config_dict(self.best_config))
            self.set_params(**params)
            self.estimator.fit(X, y)
        return self
