"""
Miscellaneous helper functions.
"""

import pandas as pd


def safe_get(d: dict, key: str, default=None):
    """Safely retrieve a key from a dict."""
    return d.get(key, default)


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of dicts (JSON-friendly)."""
    return df.to_dict(orient="records")
