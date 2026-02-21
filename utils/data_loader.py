"""
utils/data_loader.py
────────────────────
Loads and caches all three Excel sheets from the dataset file.
Sheet names:
  - Exporter_LiveSignals_v5_Updated
  - Global_News_LiveSignals_Updated
  - Importer_LiveSignals_v5_Updated
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from utils.logger import logger

SHEET_EXPORTERS = "Exporter_LiveSignals_v5_Updated"
SHEET_NEWS      = "Global_News_LiveSignals_Updated"
SHEET_IMPORTERS = "Importer_LiveSignals_v5_Updated"


@lru_cache(maxsize=1)
def load_all_sheets() -> dict[str, pd.DataFrame]:
    """
    Read all three sheets from the Excel workbook and return as a dict.
    Results are cached so the file is only read once per process.
    """
    settings = get_settings()
    excel_path: Path = settings.excel_path

    if not excel_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {excel_path}. "
            "Place EXIM_DatasetAlgo_Hackathon.xlsx inside the ./data/ folder."
        )

    logger.info(f"Loading Excel workbook from {excel_path}")

    sheets = {
        "exporters": pd.read_excel(excel_path, sheet_name=SHEET_EXPORTERS),
        "news":      pd.read_excel(excel_path, sheet_name=SHEET_NEWS),
        "importers": pd.read_excel(excel_path, sheet_name=SHEET_IMPORTERS),
    }

    for name, df in sheets.items():
        logger.info(f"Sheet '{name}': {len(df)} rows × {len(df.columns)} columns")

    return sheets


def get_exporters() -> pd.DataFrame:
    return load_all_sheets()["exporters"]


def get_importers() -> pd.DataFrame:
    return load_all_sheets()["importers"]


def get_news_signals() -> pd.DataFrame:
    return load_all_sheets()["news"]
