"""
models/preprocessor.py
──────────────────────
Cleans and feature-engineers the Exporter and Importer DataFrames
so they are ready for the matching algorithm.

Key transformations
───────────────────
1. Normalise column names (lowercase + underscores)
2. Fill / drop missing values
3. Encode categorical features (HS codes, trade terms, currencies)
4. Normalise numeric features with MinMaxScaler
5. Build a TF-IDF matrix from free-text fields (product descriptions)
"""

import re
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from utils.logger import logger

warnings.filterwarnings("ignore")


# ── internal helpers ──────────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case every column name."""
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")
        for c in df.columns
    ]
    return df


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _scale_columns(
    df: pd.DataFrame, cols: list[str], scaler: MinMaxScaler | None = None
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    cols = [c for c in cols if c in df.columns]
    if scaler is None:
        scaler = MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df, scaler


# ── public API ────────────────────────────────────────────────────────────────

NUMERIC_EXPORT_COLS = [
    "export_volume_usd", "monthly_capacity_tons", "reliability_score",
    "price_per_unit", "lead_time_days", "compliance_score",
]

NUMERIC_IMPORT_COLS = [
    "import_volume_usd", "monthly_demand_tons", "budget_per_unit",
    "max_lead_time_days", "compliance_requirement_score",
]

TEXT_COLS = ["product_description", "product_name", "trade_terms", "certification"]


def preprocess_exporters(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    logger.info("Preprocessing exporters …")
    df = _normalise_columns(df.copy())
    df = df.dropna(how="all")
    df = _coerce_numeric(df, NUMERIC_EXPORT_COLS)

    # Combine text features
    available_text = [c for c in TEXT_COLS if c in df.columns]
    df["_text_blob"] = df[available_text].fillna("").agg(" ".join, axis=1)

    df, scaler = _scale_columns(df, NUMERIC_EXPORT_COLS)
    logger.info(f"Exporters preprocessed: {len(df)} rows")
    return df, scaler


def preprocess_importers(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    logger.info("Preprocessing importers …")
    df = _normalise_columns(df.copy())
    df = df.dropna(how="all")
    df = _coerce_numeric(df, NUMERIC_IMPORT_COLS)

    available_text = [c for c in TEXT_COLS if c in df.columns]
    df["_text_blob"] = df[available_text].fillna("").agg(" ".join, axis=1)

    df, scaler = _scale_columns(df, NUMERIC_IMPORT_COLS)
    logger.info(f"Importers preprocessed: {len(df)} rows")
    return df, scaler


def build_tfidf_matrix(
    exporters: pd.DataFrame,
    importers: pd.DataFrame,
) -> Tuple[object, object, TfidfVectorizer]:
    """
    Fit a single TF-IDF vocabulary on the union of exporter + importer text,
    then return separate sparse matrices for each side.
    """
    vectorizer = TfidfVectorizer(
        max_features=512, ngram_range=(1, 2), stop_words="english"
    )
    exp_texts = exporters["_text_blob"].fillna("").tolist()
    imp_texts = importers["_text_blob"].fillna("").tolist()

    vectorizer.fit(exp_texts + imp_texts)
    exp_matrix = vectorizer.transform(exp_texts)
    imp_matrix = vectorizer.transform(imp_texts)

    logger.info(
        f"TF-IDF matrix: exporters={exp_matrix.shape}, importers={imp_matrix.shape}"
    )
    return exp_matrix, imp_matrix, vectorizer
