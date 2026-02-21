"""
Feature engineering pipeline.
Transforms raw exporter / importer data into numerical feature vectors.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureEngineer:
    """Build feature matrices from raw DataFrames."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}

    def transform(self, df: pd.DataFrame, categorical_cols: list[str], numerical_cols: list[str]) -> np.ndarray:
        """Encode categoricals and scale numericals, returning a combined feature matrix."""
        encoded_parts: list[np.ndarray] = []

        for col in categorical_cols:
            le = LabelEncoder()
            encoded_parts.append(le.fit_transform(df[col].astype(str)).reshape(-1, 1))
            self.label_encoders[col] = le

        if numerical_cols:
            scaled = self.scaler.fit_transform(df[numerical_cols].fillna(0))
            encoded_parts.append(scaled)

        return np.hstack(encoded_parts) if encoded_parts else np.array([])
