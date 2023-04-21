[![build status](https://github.com/fstermann/autoibc/actions/workflows/main.yml/badge.svg)](https://github.com/fstermann/autoibc/actions/workflows/main.yml)
[![LMU: Munich](https://img.shields.io/badge/LMU-Munich-009440.svg)](https://www.en.statistik.uni-muenchen.de/index.html)
![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)


# `autoibc` <br> Automated Machine Learning for Imbalanced Binary Classification

This repository contains my use case submission for the course *Automated Machine Learning* held by Dr. Janek Thomas and Lennart Schneider at the LMU Munich in the winter term 2022/23.

The objective of this use case was to build a automated machine learning tool for imbalanced binary classification.

# Installation

The requirements and additional information are specified in [setup.cfg](setup.cfg).

Clone this repository

```bash
git clone https://github.com/fstermann/autoibc
```
Install the package with pip
```bash
pip install .
```
or if you want to install the package with all optional dependencies to produce visualizations

```bash
pip install -e .[viz]
# pip install -e ".[viz]" # Escape for zsh
```


## Swig

You might need to install the latest version of [swig](https://www.swig.org/) in order to install `pyrfr`.

# Usage

The basic AutoIBC class can be used like any `sklearn` estimator.
As such, you can pass it into the `cross_val_score` function to cross validate the system.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from autoibc import AutoIBC
from autoibc.data import Dataset

# Instantiate the tool
auto_ibc = AutoIBC()

# Load the dataset
X, y = Dataset.from_openml(idx).to_numpy()

# Cross validate
cv = StratifiedKFold(n_splits=3)
cross_val_score(auto_ibc, X, y, scoring="balanced_accuracy", cv=cv)
```

To control variables such as inner cross validation splits, or the number of trials, you can pass these parameter to `cross_val_score` with the `fit_params` argument.

```python
cross_val_score(
    auto_ibc,
    ...,
    fit_params=dict(
        n_trials=100,
        cv_splits=5, # Number of inner cross validation splits
        outer_cv=True, # Tells the system to save the runs for each outer fold
        run_name="my-custom-run",
        seed=42,
    )
)
```


## Manual Configuration

You can also configure a pipeline on your own.
Simply pass in the name of the step along with a list of components to the `AutoPipeline` constructor.
Alternatively, pass in components as a dictionary, where the key is the component and the value is the weight assigned to it. The weight will be used by the optimization process.

It is also possible to instruct to try out configurations which skip the step. Simply pass in `None` as one of the components.

```python
from autoibc.components import classification, preprocessing, sampling
from autoibc.data import Dataset
from autoibc.pipeline import AutoPipeline

# Set up the pipeline with 3 steps
pipeline = AutoPipeline(
    steps=dict(
        preprocessing=[
            preprocessing.AutoSimpleImputer(),
        ],
        sampling=[
            None, # Will consider configurations where no sampling is done
            sampling.AutoSMOTE(),
            sampling.AutoSMOTETomek(),
        ],
        classification={
            classification.AutoRandomForest(): 3, # Prefer Random Forst
            classification.AutoGradientBoosting(): 1,
        },
    )
)
```

To create a new component, you can create a new class that inherits from `BaseAutoIBC` and implements the `configspace` property method.
Make sure to pass the `sklearn` model to the `model` argument of the `BaseAutoIBC` constructor.

Example for setting a new component for a Random Forest:
```python
from ConfigSpace import ConfigurationSpace
from sklearn.ensemble import RandomForestClassifier

from autoibc.base import BaseAutoIBC
from autoibc.hp import Boolean, Categorical, Float, Integer
from autoibc.util import make_configspace


class AutoRandomForest(BaseAutoIBC):
    def __init__(self) -> None:
        super().__init__(model=RandomForestClassifier)

    @property
    def configspace(self) -> ConfigurationSpace:
        return make_configspace(
            Boolean("bootstrap", default=True),
            Categorical("criterion", ["gini", "entropy"], default="gini"),
            Float("max_features", (0.0, 1.0), default=0.5),
            Integer("min_samples_leaf", (1, 20), default=1),
            name=self.name,
        )
```


# Benchmark

To compare the performance of the system on the given benchmark datasets against a random forest baseline, run:
```bash
python -m benchmark
```

The benchmark has been run on Google Colab.
To avoid dependecy conflicts between tensorflow, you might need to run
```bash
!pip install numpy~=1.23.0
```
after installation of the package.

## Datasets

The following datasets from [OpenML](https://www.openml.org/) are used in the benchmark example:

| ID    | Dataset                                                              | % Small Class | # Features | # Observations |
| :---- | :------------------------------------------------------------------- | ------------: | ---------: | -------------: |
| 976   | [JapaneseVowels](https://www.openml.org/search?type=data&id=976)     |          0.16 |         15 |           9961 |
| 980   | [optdigits](https://www.openml.org/search?type=data&id=980)          |          0.10 |         65 |           5620 |
| 1002  | [ipums_la_98-small](https://www.openml.org/search?type=data&id=1002) |          0.10 |         56 |           7485 |
| 1018  | [ipums_la_99-small](https://www.openml.org/search?type=data&id=1018) |          0.06 |         57 |           8844 |
| 1019  | [pendigits](https://www.openml.org/search?type=data&id=1019)         |          0.10 |         17 |          10992 |
| 1021  | [page-blocks](https://www.openml.org/search?type=data&id=1021)       |          0.10 |         11 |           5473 |
| 1040  | [sylva_prior](https://www.openml.org/search?type=data&id=1040)       |          0.06 |        109 |          14395 |
| 1053  | [jm1](https://www.openml.org/search?type=data&id=1053)               |          0.19 |         22 |          10885 |
| 1116  | [musk](https://www.openml.org/search?type=data&id=1116)              |          0.15 |        170 |           6598 |
| 41160 | [rl](https://www.openml.org/search?type=data&id=41160)               |          0.16 |         23 |          31406 |


## Visualization

Visulations of the benchmark results can be found in the [visualization](notebooks/visualization.ipynb) notebook.


# Additional Packages

The following packages are used in the implementation:
- [`openml`](https://github.com/openml/openml-python) Python API for OpenML
- [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) Machine learning library
- [`imbalanced-learn`](https://github.com/scikit-learn-contrib/imbalanced-learn) Extension of `scikit-learn` to handle imbalanced datasets
- [`smac`](https://github.com/automl/SMAC3) Bayesian optimization for hyperparameter tuning
- Visualization
    - [`matplotlib`](https://github.com/matplotlib/matplotlib) Plotting library
