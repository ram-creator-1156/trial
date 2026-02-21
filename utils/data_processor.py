"""
utils/data_processor.py
───────────────────────
EXIMDataProcessor — Robust preprocessing pipeline for the three sheets of
EXIM_DatasetAlgo_Hackathon.xlsx:

  • Exporter_LiveSignals_v5_Updated
  • Global_News_LiveSignals_Updated
  • Importer_LiveSignals_v5_Updated

Key responsibilities
────────────────────
1. Load all three sheets with graceful error handling.
2. Inspect and report data quality issues.
3. Impute missing values (median for numerics, 'Unknown' for text).
4. Normalise messy text: country names, product categories, HS codes.
5. Extract "high-intent signals" from the news sheet (tariff cuts,
   expansion announcements, new trade agreements, sanctions, etc.).
6. Return clean, structured DataFrames ready for the matchmaking engine.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── sheet names ───────────────────────────────────────────────────────────────
SHEET_EXPORTERS = "Exporter_LiveSignals_v5_Updated"
SHEET_NEWS      = "Global_News_LiveSignals_Updated"
SHEET_IMPORTERS = "Importer_LiveSignals_v5_Updated"

# ── high-intent signal keyword groups ────────────────────────────────────────
_SIGNAL_GROUPS: dict[str, list[str]] = {
    "tariff_reduction":   ["reduc", "cut tariff", "lower tariff", "tariff free",
                           "zero tariff", "duty free", "waiv"],
    "trade_agreement":    ["trade deal", "trade agreement", "fta", "bilateral",
                           "memorandum of understanding", "mou", "framework"],
    "expansion":          ["expand", "expansion", "new market", "launch",
                           "open office", "set up", "invest", "capacity"],
    "demand_surge":       ["boom", "surge", "spike", "record demand", "shortage",
                           "supply gap", "high demand", "bullish"],
    "sanction_risk":      ["sanction", "ban", "embargo", "restrict", "blacklist",
                           "deny", "prohibit"],
    "logistics_positive": ["new route", "new port", "shipping lane", "hub",
                           "logistics upgrade", "air freight", "improved transit"],
    "policy_positive":    ["incentive", "subsidy", "export promotion", "liberali",
                           "deregul", "reform"],
    "negative_risk":      ["war", "conflict", "geopolit", "disruption",
                           "force majeure", "strike", "blockade", "flood",
                           "earthquake", "pandemic"],
}

# Canonical country name mapping for common dirty variants
_COUNTRY_ALIASES: dict[str, str] = {
    "usa": "United States", "u.s.a": "United States", "u.s": "United States",
    "united states of america": "United States", "america": "United States",
    "uk": "United Kingdom", "u.k": "United Kingdom", "great britain": "United Kingdom",
    "uae": "United Arab Emirates", "u.a.e": "United Arab Emirates",
    "s. korea": "South Korea", "republic of korea": "South Korea",
    "n. korea": "North Korea", "dprk": "North Korea",
    "pr china": "China", "peoples republic of china": "China", "prc": "China",
    "czechia": "Czech Republic", "slovak republic": "Slovakia",
    "russia": "Russia", "russian federation": "Russia",
    "iran": "Iran", "islamic republic of iran": "Iran",
    "Tanzania": "Tanzania", "united republic of tanzania": "Tanzania",
    "vietnam": "Vietnam", "viet nam": "Vietnam",
    "myanmar": "Myanmar", "burma": "Myanmar",
}


# ── return dataclass ──────────────────────────────────────────────────────────

@dataclass
class ProcessedData:
    exporters:    pd.DataFrame
    importers:    pd.DataFrame
    news:         pd.DataFrame
    news_signals: pd.DataFrame   # extracted high-intent signal rows
    quality_report: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "═══ ProcessedData Summary ═══",
            f"  Exporters  : {len(self.exporters):,} rows × {len(self.exporters.columns)} cols",
            f"  Importers  : {len(self.importers):,} rows × {len(self.importers.columns)} cols",
            f"  News       : {len(self.news):,} rows × {len(self.news.columns)} cols",
            f"  Hi-Intent  : {len(self.news_signals):,} signal rows",
        ]
        for sheet, rpt in self.quality_report.items():
            lines.append(f"\n  [{sheet}] quality")
            for k, v in rpt.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)


# ── main class ────────────────────────────────────────────────────────────────

class EXIMDataProcessor:
    """
    Full preprocessing pipeline for the EXIM hackathon dataset.

    Parameters
    ----------
    excel_path : str | Path
        Absolute or relative path to EXIM_DatasetAlgo_Hackathon.xlsx.
    verbose : bool
        Print progress messages to stdout (default True).
    """

    def __init__(self, excel_path: str | Path, verbose: bool = True) -> None:
        self.excel_path = Path(excel_path)
        self.verbose = verbose
        self._raw: dict[str, pd.DataFrame] = {}

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> ProcessedData:
        """
        Execute the full pipeline and return a ProcessedData object.
        Raises FileNotFoundError if the Excel workbook is missing.
        """
        self._load_sheets()
        quality_report: dict[str, Any] = {}

        # Process each sheet independently
        exp_df, exp_qr  = self._process_exporters(self._raw["exporters"].copy())
        imp_df, imp_qr  = self._process_importers(self._raw["importers"].copy())
        news_df, news_qr = self._process_news(self._raw["news"].copy())

        quality_report["exporters"] = exp_qr
        quality_report["importers"] = imp_qr
        quality_report["news"]      = news_qr

        # Extract high-intent signals
        signals_df = self._extract_high_intent_signals(news_df)

        result = ProcessedData(
            exporters=exp_df,
            importers=imp_df,
            news=news_df,
            news_signals=signals_df,
            quality_report=quality_report,
        )

        if self.verbose:
            print(result.summary())

        return result

    # ── sheet loader ──────────────────────────────────────────────────────────

    def _load_sheets(self) -> None:
        if not self.excel_path.exists():
            raise FileNotFoundError(
                f"Excel file not found: {self.excel_path}\n"
                "Place EXIM_DatasetAlgo_Hackathon.xlsx in the data/ folder."
            )

        self._log(f"Loading workbook: {self.excel_path}")
        try:
            xl = pd.ExcelFile(self.excel_path, engine="openpyxl")
        except Exception as exc:
            raise RuntimeError(f"Cannot open Excel file: {exc}") from exc

        available_sheets = xl.sheet_names
        self._log(f"Available sheets: {available_sheets}")

        def _load(sheet_name: str) -> pd.DataFrame:
            # Fuzzy-match sheet name in case of minor capitalisation differences
            matched = next(
                (s for s in available_sheets
                 if s.strip().lower() == sheet_name.strip().lower()),
                None,
            )
            if matched is None:
                # Partial match fallback
                matched = next(
                    (s for s in available_sheets
                     if sheet_name.split("_")[0].lower() in s.lower()),
                    None,
                )
            if matched is None:
                raise KeyError(
                    f"Sheet '{sheet_name}' not found in workbook. "
                    f"Available: {available_sheets}"
                )
            df = xl.parse(matched)
            self._log(f"  ✔ Sheet '{matched}': {len(df):,} rows × {len(df.columns)} cols")
            return df

        self._raw["exporters"] = _load(SHEET_EXPORTERS)
        self._raw["news"]      = _load(SHEET_NEWS)
        self._raw["importers"] = _load(SHEET_IMPORTERS)

    # ── exporters ─────────────────────────────────────────────────────────────

    def _process_exporters(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        self._log("Processing exporters …")
        df = self._standardise_columns(df)
        qr = self._quality_report(df, label="exporters (raw)")

        df = self._impute_missing(df)
        df = self._normalise_text_fields(df, ["country", "product_category", "company_name"])
        df = self._standardise_hs_code(df)
        df = self._cast_numeric(df, [
            "export_volume_usd", "monthly_capacity_tons", "reliability_score",
            "price_per_unit", "lead_time_days", "compliance_score",
            "years_in_business", "certifications_count",
        ])
        df = self._drop_duplicate_rows(df)

        self._log(f"  Exporters ready: {len(df):,} rows")
        return df, qr

    # ── importers ─────────────────────────────────────────────────────────────

    def _process_importers(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        self._log("Processing importers …")
        df = self._standardise_columns(df)
        qr = self._quality_report(df, label="importers (raw)")

        df = self._impute_missing(df)
        df = self._normalise_text_fields(df, ["country", "product_category", "company_name"])
        df = self._standardise_hs_code(df)
        df = self._cast_numeric(df, [
            "import_volume_usd", "monthly_demand_tons", "budget_per_unit",
            "max_lead_time_days", "compliance_requirement_score",
        ])
        df = self._drop_duplicate_rows(df)

        self._log(f"  Importers ready: {len(df):,} rows")
        return df, qr

    # ── news ──────────────────────────────────────────────────────────────────

    def _process_news(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        self._log("Processing news signals …")
        df = self._standardise_columns(df)
        qr = self._quality_report(df, label="news (raw)")

        df = self._impute_missing(df)
        df = self._normalise_text_fields(df, ["country", "source"])
        df = self._standardise_hs_code(df)

        # Parse date column if present
        date_col = self._detect_col(df, "date", "published_date", "news_date", "timestamp")
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(date_col, ascending=False)

        self._log(f"  News ready: {len(df):,} rows")
        return df, qr

    # ── high-intent signal extraction ─────────────────────────────────────────

    def _extract_high_intent_signals(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scan all text columns in the news sheet for high-intent keywords.
        Adds columns:
          - signal_flags   : comma-separated signals found (e.g. 'tariff_reduction,expansion')
          - sentiment_score: +1 (positive), -1 (negative), 0 (neutral)
          - is_high_intent : True if any signal found
        """
        self._log("Extracting high-intent signals from news …")

        if news_df.empty:
            self._log("  ⚠ News DataFrame is empty, skipping signal extraction")
            return pd.DataFrame()

        # Concatenate all string columns into one searchable blob per row
        text_cols = news_df.select_dtypes(include="object").columns.tolist()
        news_df["_text_blob"] = (
            news_df[text_cols].fillna("").apply(
                lambda row: " ".join(str(v) for v in row), axis=1
            ).str.lower()
        )

        # Tag each row with matching signal groups
        positive_groups = {
            "tariff_reduction", "trade_agreement", "expansion",
            "demand_surge", "logistics_positive", "policy_positive",
        }
        negative_groups = {"sanction_risk", "negative_risk"}

        def _tag(text: str) -> tuple[str, float]:
            flags: list[str] = []
            for group, keywords in _SIGNAL_GROUPS.items():
                if any(kw in text for kw in keywords):
                    flags.append(group)
            if not flags:
                return "", 0.0
            pos = sum(1 for f in flags if f in positive_groups)
            neg = sum(1 for f in flags if f in negative_groups)
            sentiment = (pos - neg) / (pos + neg + 1e-9)
            return ",".join(flags), round(sentiment, 4)

        tags = news_df["_text_blob"].apply(_tag)
        news_df["signal_flags"]    = tags.apply(lambda t: t[0])
        news_df["sentiment_score"] = tags.apply(lambda t: t[1])
        news_df["is_high_intent"]  = news_df["signal_flags"] != ""

        # Clean up helper column
        news_df.drop(columns=["_text_blob"], inplace=True, errors="ignore")

        signal_df = news_df[news_df["is_high_intent"]].reset_index(drop=True)
        self._log(
            f"  High-intent signals found: {len(signal_df):,} / {len(news_df):,} rows"
        )

        # Per-signal-type summary
        if not signal_df.empty:
            for grp in _SIGNAL_GROUPS:
                count = signal_df["signal_flags"].str.contains(grp, na=False).sum()
                if count:
                    self._log(f"    [{grp}] → {count} rows")

        return signal_df

    # ── shared transforms ─────────────────────────────────────────────────────

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """snake_case column names; strip leading/trailing whitespace from values."""
        df.columns = [
            re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")
            for c in df.columns
        ]
        # Strip whitespace in all string cells
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "": np.nan})
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - Numeric columns  → impute with column median
        - Categorical cols → fill with 'Unknown'
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        if num_cols:
            imp = SimpleImputer(strategy="median")
            df[num_cols] = imp.fit_transform(df[num_cols])

        for col in cat_cols:
            na_count = df[col].isna().sum()
            if na_count:
                df[col].fillna("Unknown", inplace=True)

        return df

    def _normalise_text_fields(
        self, df: pd.DataFrame, preferred_cols: list[str]
    ) -> pd.DataFrame:
        """
        Apply text standardisation to specific columns if they exist.
        Extends to any column whose name contains these keywords.
        """
        target_cols = [
            c for c in df.columns
            if any(kw in c for kw in preferred_cols)
        ]
        for col in target_cols:
            df[col] = df[col].apply(self._clean_text_value)
            if "country" in col:
                df[col] = df[col].apply(self._canonicalize_country)
            elif "hs" not in col and "code" not in col:
                df[col] = df[col].apply(self._title_case_clean)
        return df

    @staticmethod
    def _clean_text_value(val: Any) -> str:
        if pd.isna(val) or val in ("Unknown", "nan", "none", ""):
            return "Unknown"
        return str(val).strip()

    @staticmethod
    def _title_case_clean(val: str) -> str:
        if val == "Unknown":
            return val
        return re.sub(r"\s+", " ", val).title()

    @staticmethod
    def _canonicalize_country(val: str) -> str:
        """Map dirty country variants to a canonical form."""
        if val == "Unknown":
            return val
        key = val.lower().strip().rstrip(".")
        return _COUNTRY_ALIASES.get(key, val.strip().title())

    def _standardise_hs_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect any HS-code column and normalise to a zero-padded 6-digit string.
        e.g.  ' 8471.3 ' → '847130', '847' → '847000'
        """
        hs_col = self._detect_col(df, "hs_code", "hscode", "hs code", "hs", "tariff_code")
        if hs_col is None:
            return df

        def _fix_hs(val: Any) -> str:
            if pd.isna(val) or str(val).strip() in ("", "Unknown", "nan"):
                return "000000"
            raw = re.sub(r"[^\d]", "", str(val))  # keep digits only
            return raw.ljust(6, "0")[:6]           # pad / truncate to 6 digits

        df[hs_col] = df[hs_col].apply(_fix_hs)
        return df

    def _cast_numeric(
        self, df: pd.DataFrame, cols: list[str]
    ) -> pd.DataFrame:
        """Coerce known-numeric columns gracefully, filling failures with 0."""
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df

    @staticmethod
    def _drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        diff = before - len(df)
        if diff:
            print(f"    Removed {diff} exact duplicate row(s)")
        return df.reset_index(drop=True)

    # ── reporting helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _quality_report(df: pd.DataFrame, label: str = "") -> dict:
        total = len(df)
        missing = df.isna().sum()
        pct_missing = (missing / total * 100).round(2) if total else missing * 0
        report = {
            "total_rows":      total,
            "total_cols":      len(df.columns),
            "duplicate_rows":  int(df.duplicated().sum()),
            "missing_by_col":  {
                col: f"{int(cnt)} ({pct_missing[col]:.1f}%)"
                for col, cnt in missing.items()
                if cnt > 0
            },
        }
        return report

    @staticmethod
    def _detect_col(df: pd.DataFrame, *candidates: str) -> str | None:
        """Return the first column name that matches any of the candidates (case-insensitive)."""
        for cand in candidates:
            match = next(
                (c for c in df.columns if cand.lower().replace(" ", "_") in c.lower()),
                None,
            )
            if match:
                return match
        return None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ── convenience function ──────────────────────────────────────────────────────

def load_and_process(excel_path: str | Path, verbose: bool = True) -> ProcessedData:
    """
    One-liner convenience wrapper.

    Example
    -------
    >>> from utils.data_processor import load_and_process
    >>> data = load_and_process("data/EXIM_DatasetAlgo_Hackathon.xlsx")
    >>> data.exporters.head()
    """
    return EXIMDataProcessor(excel_path, verbose=verbose).run()


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/EXIM_DatasetAlgo_Hackathon.xlsx"
    print(f"\nRunning EXIMDataProcessor on: {path}\n{'─'*60}")

    try:
        result = load_and_process(path)
        print("\n── Sample: Exporters (first 3 rows) ──")
        print(result.exporters.head(3).to_string())
        print("\n── Sample: High-Intent Signals (first 5 rows) ──")
        if not result.news_signals.empty:
            cols_to_show = [
                c for c in ["country", "signal_flags", "sentiment_score", "is_high_intent"]
                if c in result.news_signals.columns
            ]
            print(result.news_signals[cols_to_show].head(5).to_string())
        else:
            print("  (no signals found)")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
