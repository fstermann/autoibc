from __future__ import annotations

from autoibc.components import classification
from autoibc.components import preprocessing
from autoibc.components import sampling
from autoibc.pipeline import AutoPipeline


class AutoIBC(AutoPipeline):
    def __init__(self) -> None:
        super().__init__(
            imputation=[
                preprocessing.AutoSimpleImputer(),
                preprocessing.AutoKNNImputer(),
            ],
            scaling=[
                preprocessing.AutoStandardScaler(),
                None,
            ],
            sampling=[
                sampling.AutoSMOTE(),
                sampling.AutoSMOTEENN(),
                sampling.AutoSMOTETomek(),
            ],
            classification=[
                classification.AutoRandomForest(),
                classification.AutoGradientBoosting(),
                classification.AutoSGD(),
            ],
        )
