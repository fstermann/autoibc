from __future__ import annotations

from autoibc.components import classification
from autoibc.components import preprocessing
from autoibc.components import sampling
from autoibc.pipeline import AutoPipeline


class AutoIBC(AutoPipeline):
    def __init__(self) -> None:
        super().__init__(
            steps=dict(
                imputation=[
                    preprocessing.AutoSimpleImputer(),
                    preprocessing.AutoKNNImputer(),
                ],
                scaling=[
                    None,
                    preprocessing.AutoStandardScaler(),
                ],
                sampling=[
                    None,
                    sampling.AutoSMOTE(),
                    sampling.AutoSMOTEENN(),
                    sampling.AutoSMOTETomek(),
                ],
                classification={
                    classification.AutoRandomForest(): 10,
                    classification.AutoGradientBoosting(): 1,
                    classification.AutoSGD(): 10,
                },  # RandomForest and SGD are way faster, prioritize them
            ),
        )


class SimplePipeline(AutoPipeline):
    def __init__(self) -> None:
        super().__init__(
            steps=dict(
                imputation=[
                    preprocessing.AutoSimpleImputer(),
                ],
                scaling=[
                    preprocessing.AutoStandardScaler(),
                ],
                sampling=[
                    sampling.AutoSMOTE(),
                ],
                classification=[
                    classification.AutoRandomForest(),
                ],
            ),
        )


class RFPipeline(AutoPipeline):
    def __init__(self) -> None:
        super().__init__(
            steps=dict(
                classification=[
                    classification.AutoRandomForest(),
                ],
            ),
        )
