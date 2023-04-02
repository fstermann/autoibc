from __future__ import annotations

from ConfigSpace import ConfigurationSpace
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from autoibc.base import BaseAutoIBC
from autoibc.hp import Boolean
from autoibc.hp import Categorical
from autoibc.hp import Float
from autoibc.hp import Integer
from autoibc.util import make_configspace

# Parameters adaped from Auto-sklearn 2.0


class AutoRandomForest(BaseAutoIBC):
    """AutoIBC model for RandomForestClassifier."""

    def __init__(self) -> None:
        super().__init__(model=RandomForestClassifier)

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Boolean("bootstrap", default=True),
            Categorical("criterion", ["gini", "entropy"], default="gini"),
            Float("max_features", (0.0, 1.0), default=0.5),
            Integer("min_samples_leaf", (1, 20), default=1),
            Integer("min_samples_split", (2, 20), default=2),
            Categorical(
                "class_weight",
                ["balanced", "balanced_subsample", "None"],
                default="None",
            ),
            name=self.name,
        )


class AutoGradientBoosting(BaseAutoIBC):
    """AutoIBC model for HistGradientBoostingClassifier."""

    def __init__(self) -> None:
        super().__init__(model=HistGradientBoostingClassifier)

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Float("l2_regularization", (1e-10, 1.0), default=1e-10, log=True),
            Float("learning_rate", (0.01, 1.0), default=0.1, log=True),
            Integer("max_leaf_nodes", (3, 2047), default=31, log=True),
            Integer("min_samples_leaf", (1, 200), default=20, log=True),
            Integer("n_iter_no_change", (1, 20), default=10),
            Float("validation_fraction", (0.01, 0.4), default=0.1),
            name=self.name,
        )


class AutoSGD(BaseAutoIBC):
    """AutoIBC model for SGDClassifier."""

    def __init__(self) -> None:
        super().__init__(model=SGDClassifier)

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Float("alpha", (1e-7, 0.1), default=0.0001, log=True),
            Boolean("average", default=False),
            Float("epsilon", (1e-5, 0.1), default=0.0001, log=True),
            Float("eta0", (1e-7, 0.1), default=0.01, log=True),
            Float("l1_ratio", (1e-9, 1.0), default=0.15, log=True),
            Categorical(
                "learning_rate",
                ["optimal", "invscaling", "constant"],
                default="invscaling",
            ),
            Categorical(
                "loss",
                ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
                default="log_loss",
            ),
            Categorical("penalty", ["l1", "l2", "elasticnet"], default="l2"),
            Float("power_t", (1e-5, 1.0), default=0.5, log=True),
            Float("tol", (1e-5, 0.1), default=0.0001, log=True),
            Categorical(
                "class_weight",
                ["balanced", "None"],
                default="None",
            ),
            name=self.name,
        )
