from __future__ import annotations

from typing import Any

from ConfigSpace import ConfigurationSpace
from imblearn.base import SamplerMixin
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

from autoibc.base import BaseAutoIBC
from autoibc.hp import Integer
from autoibc.util import make_configspace


class BaseAutoSampler(BaseAutoIBC, SamplerMixin):
    def _fit_resample(self, X, y, **fit_params):
        return self.estimator._fit_resample(X, y, **fit_params)


class AutoSMOTE(BaseAutoSampler):
    """AutoIBC model for SMOTE."""

    def __init__(self) -> None:
        super().__init__(estimator=SMOTE())

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Integer("k_neighbors", (1, 25), default=5),
            name=self.name,
        )


class AutoSMOTEENN(BaseAutoSampler):
    """AutoIBC model for SMOTEENN."""

    def __init__(self) -> None:
        super().__init__(estimator=SMOTEENN())

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Integer("k_neighbors", (1, 25), default=5),
            Integer("n_neighbors", (1, 25), default=3),
            name=self.name,
        )

    def set_params(self, **params: Any) -> AutoSMOTEENN:
        params = self._unprepare_params(**params)
        k_neighbors = params.pop("k_neighbors")
        n_neighbors = params.pop("n_neighbors")
        return super().set_params(
            smote=SMOTE(k_neighbors=k_neighbors),
            enn=EditedNearestNeighbours(n_neighbors=n_neighbors),
            **params,
        )


class AutoSMOTETomek(BaseAutoSampler):
    """AutoIBC model for SMOTETomek."""

    def __init__(self) -> None:
        super().__init__(estimator=SMOTETomek())

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Integer("k_neighbors", (1, 25), default=5),
            name=self.name,
        )

    def set_params(self, **params: Any) -> AutoSMOTETomek:
        params = self._unprepare_params(**params)
        k_neighbors = params.pop("k_neighbors")
        return super().set_params(
            smote=SMOTE(k_neighbors=k_neighbors),
            **params,
        )
