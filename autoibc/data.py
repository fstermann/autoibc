from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openml
import pandas as pd
from openml import OpenMLDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize


@dataclass
class Dataset:
    name: str
    id: int
    features: pd.DataFrame
    labels: pd.DataFrame
    openml: OpenMLDataset
    encoders: dict[str, LabelEncoder]

    @classmethod
    def from_openml(cls, id: int) -> Dataset:
        """
        Processes an binary classification OpenMLDataset into its features and targets.

        Args:
            id (int): The OpenML dataset id

        Returns:
            The processed dataset
        """
        dataset = openml.datasets.get_dataset(id)
        target = dataset.default_target_attribute
        data, _, _, _ = dataset.get_data()

        assert isinstance(data, pd.DataFrame)

        # Process the features and turn all categorical columns into ints
        features = data.drop(columns=target)
        encoders: dict[str, LabelEncoder] = {}

        for name, col in features.items():
            if col.dtype in ["object", "category", "string"]:
                encoder = LabelEncoder()
                features[name] = encoder.fit_transform(col)
                encoders[name] = encoder

        labels = data[target]

        return cls(
            name=dataset.name,
            id=id,
            features=features,
            labels=labels,
            openml=dataset,
            encoders=encoders,
        )

    def _binarized_labels(self) -> np.ndarray:
        """Binarize labels of the dataset.

        This converts the labels such that the most frequent label is 0 and the
        least frequent label is 1.

        Returns:
            np.ndarray: Binarized labels
        """
        y = self.labels.to_numpy()
        counts = dict(zip(*np.unique(y, return_counts=True)))
        assert len(counts) == 2, "Dataset is not binary classification"
        neg, pos = max(counts, key=counts.get), min(counts, key=counts.get)  # type: ignore # noqa: E501
        return label_binarize(self.labels, classes=[neg, pos])[..., 0]

    def to_numpy(self, binarize: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Converts the dataset to numpy arrays."""
        if binarize:
            return self.features.to_numpy(), self._binarized_labels()
        return self.features.to_numpy(), self.labels.to_numpy()
