"""
Scorer module — computes composite compatibility scores.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_compatibility_score(
    exporter_vec: np.ndarray,
    importer_vec: np.ndarray,
    news_weight: float = 0.2,
    news_score: float = 0.0,
) -> float:
    """
    Compute a weighted compatibility score between an exporter and importer.

    Components:
        - Cosine similarity of feature vectors  (weight = 1 - news_weight)
        - News sentiment adjustment              (weight = news_weight)
    """
    cos_sim = cosine_similarity(
        exporter_vec.reshape(1, -1),
        importer_vec.reshape(1, -1),
    )[0][0]

    score = (1 - news_weight) * cos_sim + news_weight * news_score
    return float(np.clip(score, 0.0, 1.0))
