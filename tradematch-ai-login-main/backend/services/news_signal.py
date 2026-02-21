"""
Global news signal processing service.
"""

import pandas as pd


def enrich_with_news_signals(df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich exporter/importer data with relevant news sentiment signals."""
    # TODO: implement news-based feature enrichment
    return df
