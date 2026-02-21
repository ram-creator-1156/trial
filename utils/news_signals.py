"""
utils/news_signals.py
─────────────────────
Processes the Global_News_LiveSignals sheet to produce:
  - A mapping: HS code → news sentiment score  (float -1 … +1)
  - Banned / sanctioned country flags
These signals are blended into the final match score.
"""

import re

import numpy as np
import pandas as pd

from utils.logger import logger


# ── helpers ──────────────────────────────────────────────────────────────────

_POSITIVE_KEYWORDS = [
    "boom", "surge", "record", "growth", "demand", "opportunity",
    "bullish", "rally", "expansion", "uplift",
]
_NEGATIVE_KEYWORDS = [
    "sanction", "ban", "tariff", "restriction", "slowdown", "decline",
    "shortage", "embargo", "risk", "war",
]


def _simple_sentiment(text: str) -> float:
    """Naïve keyword-based sentiment scorer. Returns value in [-1, +1]."""
    if not isinstance(text, str):
        return 0.0
    text_lower = text.lower()
    pos = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text_lower)
    neg = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text_lower)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def build_hs_sentiment_map(news_df: pd.DataFrame) -> dict[str, float]:
    """
    Returns {hs_code: avg_sentiment} for every HS code mentioned in news.
    Tries common column name variants automatically.
    """
    # Detect text column
    text_col = next(
        (c for c in news_df.columns if re.search(r"headline|title|news|text", c, re.I)),
        None,
    )
    hs_col = next(
        (c for c in news_df.columns if re.search(r"hs.?code|hscode|hs_code", c, re.I)),
        None,
    )

    if text_col is None or hs_col is None:
        logger.warning(
            "news_signals: Could not auto-detect required columns. "
            f"Available: {list(news_df.columns)}"
        )
        return {}

    news_df = news_df[[hs_col, text_col]].copy()
    news_df["sentiment"] = news_df[text_col].apply(_simple_sentiment)
    result = news_df.groupby(hs_col)["sentiment"].mean().to_dict()

    logger.info(f"Built HS→sentiment map for {len(result)} HS codes")
    return result


def build_banned_countries(news_df: pd.DataFrame) -> set[str]:
    """
    Returns a set of country names flagged in news as sanctioned / banned.
    """
    country_col = next(
        (c for c in news_df.columns if re.search(r"country", c, re.I)),
        None,
    )
    text_col = next(
        (c for c in news_df.columns if re.search(r"headline|title|news|text", c, re.I)),
        None,
    )

    if country_col is None or text_col is None:
        return set()

    mask = news_df[text_col].str.contains(
        "|".join(["sanction", "ban", "embargo"]), case=False, na=False
    )
    banned = set(news_df.loc[mask, country_col].dropna().str.strip().str.upper())
    logger.info(f"Identified {len(banned)} potentially restricted countries from news")
    return banned
