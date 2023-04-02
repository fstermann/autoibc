# This module is meant to provide a unified interface for hyperparameters
# used in autoibc.

from __future__ import annotations

from ConfigSpace import Categorical  # noqa
from ConfigSpace import Constant as CsConstant
from ConfigSpace import Float  # noqa
from ConfigSpace import Integer  # noqa
from ConfigSpace.hyperparameters import CategoricalHyperparameter


class Constant(CsConstant):
    """Representing a constant hyperparameter in the configuration space.

    Taken from:
        https://automl.github.io/ConfigSpace/main/api/hyperparameters.html#constant
    By sampling from the configuration space each time only a single, constant
    value will be drawn from this hyperparameter.

    Args:
        name (str): Name of the hyperparameter, with which it can be accessed
        value (str, int, float): value to sample hyperparameter from
        meta (Dict, optional): Field for holding meta data provided by the user.
            Not used by the configuration space.
    """

    def __init__(
        self,
        name: str,
        value: str | int | float,
        meta: dict | None = None,
    ) -> None:
        super().__init__(name=name, value=value, meta=meta)


class Boolean(CategoricalHyperparameter):
    """Boolean hyperparameter.

    This is a wrapper around the ConfigSpace CategoricalHyperparameter
    to provide a more intuitive interface for boolean hyperparameters.

    Args:
        name (str): Name of the hyperparameter, with which it can be accessed
        default (bool, optional): Default value of the hyperparameter.
            Defaults to False.
    """

    def __init__(self, name: str, default: bool = False) -> None:
        super().__init__(
            name=name,
            choices=["True", "False"],
            default_value=str(default),
        )
