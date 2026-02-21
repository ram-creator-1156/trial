"""
Core matchmaking algorithm.
Uses scikit-learn similarity metrics to rank importer compatibility for each exporter.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class Matchmaker:
    """Rank importers by compatibility with a given exporter."""

    def __init__(self) -> None:
        self.exporter_features: pd.DataFrame | None = None
        self.importer_features: pd.DataFrame | None = None

    def fit(self, exporter_features: pd.DataFrame, importer_features: pd.DataFrame) -> None:
        """Store pre-processed feature matrices."""
        self.exporter_features = exporter_features
        self.importer_features = importer_features

    def rank(self, exporter_id: str, top_n: int = 10) -> list[dict]:
        """Return top-N importers ranked by cosine similarity to the exporter."""
        # TODO: implement once feature engineering is complete
        return []

    @staticmethod
    def compute_similarity(vec_a, vec_b):
        """Compute cosine similarity between two feature vectors."""
        return cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0][0]
