"""
Data loader service — reads the Excel dataset into Pandas DataFrames.
"""

import pandas as pd
from backend.core.config import settings

SHEET_EXPORTERS = "Exporter_LiveSignals_v5_Updated"
SHEET_NEWS = "Global_News_LiveSignals_Updated"
SHEET_IMPORTERS = "Importer_LiveSignals_v5_Updated"


def load_exporters(path: str | None = None) -> pd.DataFrame:
    """Load the Exporters sheet."""
    path = path or settings.DATA_FILE
    return pd.read_excel(path, sheet_name=SHEET_EXPORTERS)


def load_importers(path: str | None = None) -> pd.DataFrame:
    """Load the Importers sheet."""
    path = path or settings.DATA_FILE
    return pd.read_excel(path, sheet_name=SHEET_IMPORTERS)


def load_news(path: str | None = None) -> pd.DataFrame:
    """Load the Global News sheet."""
    path = path or settings.DATA_FILE
    return pd.read_excel(path, sheet_name=SHEET_NEWS)


def load_all(path: str | None = None) -> dict[str, pd.DataFrame]:
    """Load all three sheets and return as a dict."""
    path = path or settings.DATA_FILE
    return {
        "exporters": load_exporters(path),
        "importers": load_importers(path),
        "news": load_news(path),
    }
