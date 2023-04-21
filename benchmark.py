from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn import impute
from sklearn import pipeline
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from autoibc import AutoIBC
from autoibc.data import Dataset

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_ids",
    type=int,
    default=[976, 980, 1002, 1018, 1019, 1021, 1040, 1053, 1116, 41160],
    nargs="+",
    help="OpenML data ids to run on",
)
parser.add_argument(
    "--scoring",
    type=str,
    default="balanced_accuracy",
    help="Scoring metric to use for evaluation",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed to use for reproducibility",
)
parser.add_argument(
    "--n_trials",
    type=int,
    default=100,
    help="Number of trials to run for each dataset",
)
parser.add_argument("--outer_cv", type=int, default=2, help="Number of outer CV folds")
parser.add_argument("--inner_cv", type=int, default=5, help="Number of inner CV folds")

args = parser.parse_args()


# Setup Pipelines for AutoIBC and Baseline
auto_ibc = AutoIBC()
baseline = pipeline.Pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", tree.DecisionTreeClassifier()),
    ],
)


for idx in args.data_ids:
    # Load dataset from OpenML
    dataset = Dataset.from_openml(idx)
    X, y = dataset.to_numpy()
    print(f"\nRunning on {dataset.name} [{idx}]")

    # Evaluation against random forest baseline
    cv_splits = StratifiedKFold(
        n_splits=args.outer_cv,
        shuffle=True,
        random_state=args.seed,
    )

    run_name = f"autoibc-{idx}-folds{args.outer_cv}-trials{args.n_trials}"
    auto_ibc_scores = cross_val_score(
        auto_ibc,
        X,
        y,
        scoring=args.scoring,
        cv=cv_splits,
        fit_params=dict(
            n_trials=args.n_trials,
            cv_splits=args.inner_cv,
            outer_cv=True,
            run_name=run_name,
            seed=args.seed,
        ),
        n_jobs=-1,  # Parallelize folds
    )
    print("AutoIBC scores: ", np.mean(auto_ibc_scores))

    baseline_scores = cross_val_score(
        baseline,
        X,
        y,
        scoring=args.scoring,
        cv=cv_splits,
    )
    print("Baseline scores: ", np.mean(baseline_scores))

    # Save results
    results = {
        "autoibc": np.mean(auto_ibc_scores),
        "baseline": np.mean(baseline_scores),
        "autoibc_scores": auto_ibc_scores.tolist(),
        "baseline_scores": baseline_scores.tolist(),
    }
    print(results)
    with open(f"results/{run_name}/results.json", "w") as f:
        json.dump(results, f)
