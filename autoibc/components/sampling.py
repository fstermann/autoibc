from __future__ import annotations

from typing import Any

from ConfigSpace import ConfigurationSpace
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

from autoibc.base import BaseAutoIBC
from autoibc.hp import Integer
from autoibc.util import make_configspace


class AutoSMOTE(BaseAutoIBC):
    """AutoIBC model for SMOTE."""

    def __init__(self) -> None:
        super().__init__(model=SMOTE)

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Integer("k_neighbors", (1, 25), default=5),
            name=self.name,
        )


class AutoSMOTEENN(BaseAutoIBC):
    """AutoIBC model for SMOTEENN."""

    def __init__(self) -> None:
        super().__init__(model=SMOTEENN)

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Integer("k_neighbors", (1, 25), default=5),
            Integer("n_neighbors", (1, 25), default=3),
            name=self.name,
        )

    def get_model(self, config: dict[str, Any]) -> SMOTEENN:
        k_neighbors = config.pop("k_neighbors")
        n_neighbors = config.pop("n_neighbors")
        return self.model(
            smote=SMOTE(k_neighbors=k_neighbors),
            enn=EditedNearestNeighbours(n_neighbors=n_neighbors),
            **config,
        )


class AutoSMOTETomek(BaseAutoIBC):
    """AutoIBC model for SMOTETomek."""

    def __init__(self) -> None:
        super().__init__(model=SMOTETomek)

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Integer("k_neighbors", (1, 25), default=5),
            name=self.name,
        )

    def get_model(self, config: dict[str, Any]) -> SMOTETomek:
        k_neighbors = config.pop("k_neighbors")
        return self.model(
            smote=SMOTE(k_neighbors=k_neighbors),
            **config,
        )
