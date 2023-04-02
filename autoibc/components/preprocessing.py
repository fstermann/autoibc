from __future__ import annotations

from ConfigSpace import ConfigurationSpace
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from autoibc.base import BaseAutoIBC
from autoibc.hp import Boolean
from autoibc.hp import Categorical
from autoibc.hp import Integer
from autoibc.util import make_configspace


class AutoSimpleImputer(BaseAutoIBC):
    """AutoIBC model for SimpleImputer."""

    def __init__(self) -> None:
        super().__init__(model=SimpleImputer)

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Categorical(
                "strategy",
                ["mean", "median", "most_frequent"],
                default="mean",
            ),
            name=self.name,
        )


class AutoKNNImputer(BaseAutoIBC):
    """AutoIBC model for KNNImputer."""

    def __init__(self) -> None:
        super().__init__(model=KNNImputer)

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Integer("n_neighbors", (1, 25), default=5),
            name=self.name,
        )


class AutoStandardScaler(BaseAutoIBC):
    """AutoIBC model for StandardScaler."""

    def __init__(self) -> None:
        super().__init__(model=StandardScaler)

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Boolean("with_mean", default=True),
            Boolean("with_std", default=True),
            name=self.name,
        )
