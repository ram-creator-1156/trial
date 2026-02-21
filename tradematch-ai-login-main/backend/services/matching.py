"""
Matching orchestration service.
Coordinates feature engineering, scoring, and ranking.
"""

from models.matchmaker import Matchmaker


def get_top_matches(exporter_id: str, top_n: int = 10) -> list[dict]:
    """Return the top N importer matches for a given exporter."""
    # TODO: implement full pipeline
    matchmaker = Matchmaker()
    return matchmaker.rank(exporter_id, top_n=top_n)
