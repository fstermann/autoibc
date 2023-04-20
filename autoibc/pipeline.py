from __future__ import annotations

from typing import Any

from ConfigSpace import ConfigurationSpace
from imblearn.pipeline import Pipeline

from autoibc.base import BaseAutoIBC
from autoibc.hp import Categorical


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
        steps: dict[str, list[BaseAutoIBC | None] | dict[BaseAutoIBC | None, float]],
    ) -> None:
        self.auto_steps = self._validate_hp_steps(steps)
        default_steps = [
            (step.name, list(step.models.values())[0])
            for step in self.auto_steps.values()
        ]
        super().__init__(estimator=Pipeline(steps=default_steps))

    def _validate_hp_steps(self, steps) -> dict[str, AutoStep]:
        auto_steps = {}
        for step, models in steps.items():
            if not isinstance(models, (list, dict)):
                raise ValueError("Step must be a list or a dict")
            weights = [1.0 for _ in models]
            if isinstance(models, dict):
                weights = list(models.values())
                models = list(models.keys())
            auto_steps[step] = AutoStep(name=step, models=models, weights=weights)
        return auto_steps

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
        steps = self.auto_steps
        cs = ConfigurationSpace(name=self.name)
        for step in steps.values():
            cs.add_hyperparameter(step.as_hp())
            for model in step.models.values():
                if not model:
                    continue
                cs.add_configuration_space(
                    prefix=model.name + ":" + step.name,
                    delimiter="__",
                    configuration_space=model.configspace,
                    parent_hyperparameter={
                        "parent": cs[step.name],
                        "value": model.name,
                    },
                )
        return cs

    def set_params(self, **params):
        """Manual hacky way to set the parameters of the pipeline."""
        # print(
        #     " -> ".join(
        #         [params[step.name] or "None" for step in self.auto_steps.values()],
        #     ),
        # )
        new_steps = []
        for step in self.auto_steps.values():
            step_class = params[step.name]
            value = step.models[step_class] if step_class else "passthrough"
            params[step.name] = value
            new_steps.append((step.name, value))

        new_pipeline = Pipeline(steps=new_steps)
        params = self._prepare_params(**params)
        new_pipeline.set_params(**params)

        steps_with_new_params = [
            (name, (step.estimator if not isinstance(step, str) else step))
            for (name, step) in new_pipeline.steps
        ]
        self.estimator = Pipeline(steps=steps_with_new_params)
        return self

    def _prepare_params(self, **params: Any) -> dict[str, Any]:
        """Prepares the parameters for the model.

        This can be overwritten by subclasses to implement custom logic.

        Args:
            **params (Any): Parameters to prepare

        Returns:
            dict[str, Any]: The prepared parameters
        """
        params = {k.split(":")[-1]: v for k, v in params.items()}
        params = {
            (k.replace("__", "__estimator__") if "estimator" not in k else k): v
            for k, v in params.items()
        }
        return params

    def predict(self, X):
        return self.estimator.predict(X)
