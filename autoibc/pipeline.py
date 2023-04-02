from __future__ import annotations

from typing import Any

from ConfigSpace import ConfigurationSpace
from imblearn.pipeline import Pipeline

from autoibc.base import BaseAutoIBC
from autoibc.hp import Categorical
from autoibc.logging import logger


class AutoStep:
    """A step in an AutoPipeline.

    Args:
        name (str): Name of the step
        models (list[BaseAutoIBC | None]): List of models
        weights (list[float]): Weights for the models
    """

    name: str
    models: dict[str, BaseAutoIBC | None]
    weights: list[float]

    def __init__(
        self,
        name: str,
        models: list[BaseAutoIBC | None],
        weights: list[float],
    ) -> None:
        self.name = name
        model_names = [model.name if model else "None" for model in models]
        self.models = dict(zip(model_names, models))
        self.weights = weights

    def as_hp(self) -> Categorical:
        """Returns the step as a hyperparameter."""
        return Categorical(self.name, list(self.models), weights=self.weights)


class AutoPipeline(BaseAutoIBC):
    """A pipeline of AutoIBC models.

    A pipeline is a list of steps, where each step is a list of models.
    The models of each step can be provided as a list or a dictionary of
    models and their weights.

    Args:
        **steps (list[BaseAutoIBC | None] | dict[BaseAutoIBC | None, float]):
            List of models or dictionary of models and their weights
    """

    def __init__(
        self,
        **steps: list[BaseAutoIBC | None] | dict[BaseAutoIBC | None, float],
    ) -> None:
        super().__init__(model=Pipeline)

        self._steps = {}
        for step, models in steps.items():
            if not isinstance(models, (list, dict)):
                raise ValueError("Step must be a list or a dict")
            weights = [1.0 for _ in models]
            if isinstance(models, dict):
                weights = list(models.values())
                models = list(models.keys())
            self._steps[step] = AutoStep(name=step, models=models, weights=weights)

    @property
    def steps(self) -> dict[str, AutoStep]:
        """Returns the steps of the pipeline."""
        return self._steps

    @property
    def step_names(self) -> list[str]:
        """Returns the names of the steps of the pipeline."""
        return list(self._steps)

    @property
    def configspace(self) -> ConfigurationSpace:
        """Custom configspace for the pipeline.

        The configspace of the pipeline consists of a hyperparameter for each
        step and a configuration space for each model in each step.

        Returns:
            ConfigurationSpace: Configuration space for the pipeline
        """
        cs = ConfigurationSpace(name=self.name)
        for step in self.steps.values():
            cs.add_hyperparameter(step.as_hp())
            for model in step.models.values():
                if not model:
                    continue
                cs.add_configuration_space(
                    prefix=model.name,
                    configuration_space=model.configspace,
                    parent_hyperparameter={
                        "parent": cs[step.name],
                        "value": model.name,
                    },
                )
        return cs

    def get_model(self, config: dict[str, Any]) -> Any:
        """Instantiates the pipeline with the given configuration.

        For each step in the pipeline, the model name is extracted from the
        configuration and the corresponding model is instantiated.

        Args:
            config (dict[str, Any]): The configuration dictionary

        Returns:
            Any: The instantiated pipeline
        """
        all_steps = []
        for step in self.steps.values():
            model_name = config.pop(step.name)
            if model_name is None:
                continue
            model_class = step.models[model_name]
            model_params = self._filter_config(config, model_name)
            assert model_class is not None
            model = model_class.get_model(model_params)
            all_steps.append((step.name, model))

        if config:
            logger.warning(f"Unused hyperparameters: {config}")
        return self.model(all_steps)

    @staticmethod
    def _filter_config(config: dict[str, Any], name: str) -> dict[str, Any]:
        """Filters the config for a specific model name.

        Args:
            config (dict[str, Any]): The config dictionary
            name (str): The model name

        Returns:
            dict[str, Any]: The configuration dictionary for the model
        """
        filtered_config = {}
        for key in list(config):
            if key.startswith(name):
                filtered_config[key.replace(f"{name}:", "")] = config.pop(key)
        return filtered_config
