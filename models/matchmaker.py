"""
models/matchmaker.py
════════════════════
MatchmakingEngine  — Multi-criteria scoring (0–100) between Exporters and Importers.

Four Scoring Pillars
────────────────────
  Pillar 1 — Product Compatibility      30 pts
      • HS-code character overlap (prefix match)
      • TF-IDF cosine similarity on Industry descriptions
      • Certification alignment bonus

  Pillar 2 — Geography Fit              20 pts
      • Direct country match
      • Same macro-region match (continent / trade bloc)
      • State-to-region bonus for Indian exporters

  Pillar 3 — High-Intent Signals        30 pts
      • Importer-side live signals: Intent_Score, Hiring_Growth,
        Funding_Event, Engagement_Spike, DecisionMaker_Change
      • Cross-reference with processed news_signals DataFrame
        (tariff_reduction, expansion, trade_agreement bonuses)
      • Risk penalty: War_Event, StockMarket_Shock, Natural_Calamity,
        Currency_Fluctuation, Tariff_News

  Pillar 4 — Trade Frequency & Activity 20 pts
      • Avg_Order_Tons vs Exporter Quantity_Tons
      • Revenue_Size_USD comparison
      • Response_Probability, Good_Payment_History, Prompt_Response
      • Exporter Prompt_Response_Score, Good_Payment_Terms

Output
──────
  DataFrame with columns:
    Exporter_ID | Importer_Buyer_ID | Total_Score | Grade |
    P1_Product | P2_Geography | P3_Signals | P4_Activity |
    Explanation (dict serialised as JSON string)

Usage
─────
  from utils.data_processor import load_and_process
  from models.matchmaker import MatchmakingEngine

  data   = load_and_process("data/EXIM_DatasetAlgo_Hackathon.xlsx")
  engine = MatchmakingEngine(data.exporters, data.importers, data.news_signals)
  result = engine.match(top_k=10)          # global top-K
  result = engine.match_for_exporter("EXP_0001", top_k=5)
  result = engine.match_for_importer("BYR_0042", top_k=5)
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Constants & mappings
# ─────────────────────────────────────────────────────────────────────────────

PILLAR_WEIGHTS = {
    "product":    0.30,
    "geography":  0.20,
    "signals":    0.30,
    "activity":   0.20,
}

# Macro-region groupings for geography scoring
_REGION_MAP: dict[str, str] = {
    # South Asia
    "India": "South Asia", "Pakistan": "South Asia", "Bangladesh": "South Asia",
    "Sri Lanka": "South Asia", "Nepal": "South Asia", "Bhutan": "South Asia",
    # Southeast Asia
    "Vietnam": "Southeast Asia", "Thailand": "Southeast Asia", "Malaysia": "Southeast Asia",
    "Indonesia": "Southeast Asia", "Philippines": "Southeast Asia", "Singapore": "Southeast Asia",
    "Myanmar": "Southeast Asia", "Cambodia": "Southeast Asia",
    # East Asia
    "China": "East Asia", "Japan": "East Asia", "South Korea": "East Asia",
    "Taiwan": "East Asia", "Hong Kong": "East Asia",
    # Middle East
    "United Arab Emirates": "Middle East", "Saudi Arabia": "Middle East",
    "Qatar": "Middle East", "Kuwait": "Middle East", "Oman": "Middle East",
    "Bahrain": "Middle East", "Jordan": "Middle East", "Iraq": "Middle East",
    "Iran": "Middle East", "Turkey": "Middle East",
    # Europe
    "Germany": "Europe", "France": "Europe", "United Kingdom": "Europe",
    "Italy": "Europe", "Spain": "Europe", "Netherlands": "Europe",
    "Belgium": "Europe", "Poland": "Europe", "Sweden": "Europe",
    "Czech Republic": "Europe", "Austria": "Europe", "Switzerland": "Europe",
    # North America
    "United States": "North America", "Canada": "North America", "Mexico": "North America",
    # Latin America
    "Brazil": "Latin America", "Argentina": "Latin America", "Chile": "Latin America",
    "Colombia": "Latin America", "Peru": "Latin America",
    # Africa
    "Nigeria": "Africa", "Kenya": "Africa", "South Africa": "Africa",
    "Ethiopia": "Africa", "Ghana": "Africa", "Tanzania": "Africa",
    "Egypt": "Africa",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania",
}

# Indian state → export region tag
_INDIA_STATE_REGION: dict[str, str] = {
    "Maharashtra": "West India", "Gujarat": "West India", "Rajasthan": "West India",
    "Punjab": "North India", "Haryana": "North India", "Delhi": "North India",
    "Uttar Pradesh": "North India", "Uttarakhand": "North India",
    "Tamil Nadu": "South India", "Karnataka": "South India", "Kerala": "South India",
    "Andhra Pradesh": "South India", "Telangana": "South India",
    "West Bengal": "East India", "Odisha": "East India", "Bihar": "East India",
    "Jharkhand": "East India", "Assam": "Northeast India",
}

# News Signal → positive flag
_POSITIVE_SIGNAL_FLAGS = {
    "tariff_reduction", "trade_agreement", "expansion",
    "demand_surge", "logistics_positive", "policy_positive",
}
_NEGATIVE_SIGNAL_FLAGS = {"sanction_risk", "negative_risk"}

# Letter grades by score bracket
def _grade(score: float) -> str:
    if score >= 85: return "A+"
    if score >= 75: return "A"
    if score >= 65: return "B+"
    if score >= 55: return "B"
    if score >= 45: return "C"
    return "D"


# ─────────────────────────────────────────────────────────────────────────────
#  Column resolver helpers
# ─────────────────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, *candidates: str, default: Any = None) -> str | None:
    """Return the first matching column name (case-insensitive snake-match)."""
    df_cols_lower = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        if key in df_cols_lower:
            return df_cols_lower[key]
    return default


def _val(row: pd.Series, *candidates: str, default: Any = 0) -> Any:
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        for col in row.index:
            if col.lower().replace(" ", "_") == key:
                return row[col]
    return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
#  Main Engine
# ─────────────────────────────────────────────────────────────────────────────

class MatchmakingEngine:
    """
    Parameters
    ----------
    exporters_df   : cleaned exporter DataFrame (from EXIMDataProcessor)
    importers_df   : cleaned importer DataFrame (from EXIMDataProcessor)
    news_signals_df: high-intent signal rows (from EXIMDataProcessor)
                     May be empty — engine degrades gracefully.
    verbose        : print progress messages
    """

    def __init__(
        self,
        exporters_df:    pd.DataFrame,
        importers_df:    pd.DataFrame,
        news_signals_df: pd.DataFrame | None = None,
        verbose: bool = True,
    ) -> None:
        self.exp = exporters_df.reset_index(drop=True).copy()
        self.imp = importers_df.reset_index(drop=True).copy()
        self.news = (
            news_signals_df.reset_index(drop=True).copy()
            if news_signals_df is not None and not news_signals_df.empty
            else pd.DataFrame()
        )
        self.verbose = verbose

        self._log("Initialising MatchmakingEngine …")
        self._normalise_column_names()
        self._build_tfidf()
        self._build_news_index()
        self._build_numpy_arrays()   # ← pre-compute vectorised pillar arrays
        self._log(
            f"  Ready — {len(self.exp):,} exporters × {len(self.imp):,} importers "
            f"| {len(self.news):,} news signal rows"
        )

    # ── setup ─────────────────────────────────────────────────────────────────

    def _normalise_column_names(self) -> None:
        """Lower-snake all column names for uniform access."""
        def _snake(df: pd.DataFrame) -> pd.DataFrame:
            df.columns = [
                re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")
                for c in df.columns
            ]
            return df
        self.exp  = _snake(self.exp)
        self.imp  = _snake(self.imp)
        if not self.news.empty:
            self.news = _snake(self.news)

    def _build_tfidf(self) -> None:
        """Fit a shared TF-IDF vocabulary on exporter + importer Industry text."""
        exp_text_col = _col(self.exp, "industry", "product_category", "product_name")
        imp_text_col = _col(self.imp, "industry", "product_category", "product_name")

        exp_texts = (
            self.exp[exp_text_col].fillna("").astype(str).tolist()
            if exp_text_col else [""] * len(self.exp)
        )
        imp_texts = (
            self.imp[imp_text_col].fillna("").astype(str).tolist()
            if imp_text_col else [""] * len(self.imp)
        )

        self._vectorizer = TfidfVectorizer(
            max_features=256, ngram_range=(1, 2), stop_words="english"
        )
        self._vectorizer.fit(exp_texts + imp_texts)
        self._exp_tfidf = self._vectorizer.transform(exp_texts)
        self._imp_tfidf = self._vectorizer.transform(imp_texts)

        self._log(
            f"  TF-IDF matrix: exp={self._exp_tfidf.shape}, imp={self._imp_tfidf.shape}"
        )

    def _build_news_index(self) -> None:
        """
        Build a lookup: industry → [list of signal_flags] for fast cross-referencing.
        """
        self._news_index: dict[str, list[str]] = {}
        if self.news.empty:
            return

        industry_col = _col(self.news, "affected_industry", "industry", "event_type")
        signal_col   = _col(self.news, "signal_flags")

        if industry_col and signal_col:
            for _, row in self.news.iterrows():
                ind = str(row.get(industry_col, "")).strip().lower()
                flags = str(row.get(signal_col, ""))
                if ind and flags:
                    self._news_index.setdefault(ind, []).append(flags)

        self._log(f"  News index built: {len(self._news_index)} unique industries")

    def _build_numpy_arrays(self) -> None:
        """
        Pre-compute per-exporter and per-importer NumPy vectors for all pillars
        that can be vectorised. Stored as 1-D arrays indexed by row position.

        Exporter arrays (shape = n_exp):
          _e_cert, _e_ind_tokens, _e_region, _e_state_bonus,
          _e_intent, _e_hiring, _e_linkedin,
          _e_supply_tons, _e_rev, _e_pay_terms, _e_resp_score, _e_salesnav,
          _e_ids

        Importer arrays (shape = n_imp):
          _i_cert, _i_industry_str, _i_region,
          _i_intent, _i_hiring, _i_funding, _i_eng, _i_dm,
          _i_war, _i_stock, _i_calamity, _i_currency, _i_tariff,
          _i_avg_order, _i_rev, _i_good_pay, _i_prompt, _i_resp_prob,
          _i_news_bonus,
          _i_ids
        """
        self._log("  Building vectorised NumPy arrays …")
        E = self.exp
        I = self.imp

        def _arr(df, *cols, default=0.0):
            """Extract a writeable float64 array for the first matching column."""
            for c in cols:
                key = c.lower().replace(" ", "_")
                match = next((col for col in df.columns
                              if col.lower().replace(" ", "_") == key), None)
                if match is not None:
                    return pd.to_numeric(df[match], errors="coerce").fillna(default).to_numpy(dtype=np.float64).copy()
            return np.full(len(df), default, dtype=np.float64)

        def _str_arr(df, *cols, default=""):
            for c in cols:
                key = c.lower().replace(" ", "_")
                match = next((col for col in df.columns
                              if col.lower().replace(" ", "_") == key), None)
                if match is not None:
                    return df[match].fillna(default).astype(str).to_numpy().copy()
            return np.full(len(df), default, dtype=object)

        # ── Exporter arrays ───────────────────────────────────────────────────
        self._e_ids         = _str_arr(E, "exporter_id", "exp_id", "record_id")
        self._e_cert        = _str_arr(E, "certification")
        self._e_ind         = _str_arr(E, "industry", "product_category")
        self._e_state       = _str_arr(E, "state")
        self._e_intent      = np.clip(_arr(E, "intent_score") / 10.0, 0, 1)
        self._e_hiring      = np.clip(_arr(E, "hiring_signal"), 0, 1)
        self._e_linkedin    = np.clip(_arr(E, "linkedin_activity"), 0, 1)
        self._e_supply_tons = np.maximum(
            _arr(E, "quantity_tons"),
            _arr(E, "manufacturing_capacity_tons")
        )
        self._e_rev         = _arr(E, "revenue_size_usd")
        self._e_pay_terms   = np.clip(_arr(E, "good_payment_terms"), 0, 1)
        self._e_resp_score  = np.clip(_arr(E, "prompt_response_score"), 0, 1)
        raw_salesnav        = _arr(E, "salesnav_profileviews")
        self._e_salesnav    = np.where(raw_salesnav > 1, np.clip(raw_salesnav / 1000.0, 0, 1), raw_salesnav)

        # Exporter signals composite (shape n_exp)
        self._e_signal_vec  = np.clip(
            (self._e_intent + self._e_hiring + self._e_linkedin) / 3.0, 0, 1
        )

        # Region lookup for each exporter (all are Indian exporters → South Asia)
        # If a 'country' column exists use it; otherwise fall back to State→India
        exp_country_arr = _str_arr(E, "country", default="India")
        # If all blanks, infer India from state column
        exp_country_arr = np.where(
            (exp_country_arr == "") | (exp_country_arr == "Unknown"),
            "India",
            exp_country_arr,
        )
        self._e_region = np.array(
            [_REGION_MAP.get(c, "South Asia") for c in exp_country_arr]
        )
        self._e_state_bonus = np.array(
            [0.05 if s in _INDIA_STATE_REGION else 0.0 for s in self._e_state]
        )

        # ── Importer arrays ───────────────────────────────────────────────────
        self._i_ids       = _str_arr(I, "buyer_id", "importer_id", "record_id")
        self._i_cert      = _str_arr(I, "certification")
        self._i_ind       = _str_arr(I, "industry", "product_category")
        imp_country_arr   = _str_arr(I, "country", default="Unknown")
        self._i_region    = np.array(
            [_REGION_MAP.get(c, "Unknown") for c in imp_country_arr]
        )
        self._i_intent    = np.clip(_arr(I, "intent_score") / 10.0, 0, 1)
        self._i_hiring    = np.clip(_arr(I, "hiring_growth"), 0, 1)
        self._i_funding   = np.clip(_arr(I, "funding_event"), 0, 1)
        self._i_eng       = np.clip(_arr(I, "engagement_spike"), 0, 1)
        self._i_dm        = np.clip(_arr(I, "decisionmaker_change"), 0, 1)
        self._i_war       = _arr(I, "war_event")
        self._i_stock     = _arr(I, "stockmarket_shock")
        self._i_calamity  = _arr(I, "natural_calamity")
        self._i_currency  = _arr(I, "currency_fluctuation")
        self._i_tariff    = _arr(I, "tariff_news")
        self._i_avg_order = _arr(I, "avg_order_tons")
        self._i_rev       = _arr(I, "revenue_size_usd")
        raw_gpay          = _arr(I, "good_payment_history")
        self._i_good_pay  = np.clip(raw_gpay, 0, 1)
        self._i_prompt    = np.clip(_arr(I, "prompt_response"), 0, 1)
        raw_resp          = _arr(I, "response_probability")
        self._i_resp_prob = np.where(raw_resp > 1, raw_resp / 100.0, raw_resp)

        # Pre-compute news bonus per importer industry (float array)
        self._i_news_bonus = np.array(
            [self._news_signal_bonus(ind.lower().strip()) for ind in self._i_ind]
        )

        # Geography corridor lookup pre-computed as float matrix slice is too large;
        # use a string→float dict instead (fast dict lookup per query)
        self._corridor: dict[tuple[str,str], float] = {}
        for r1 in set(self._e_region):
            for r2 in set(self._i_region):
                key = (r1, r2)
                if r1 == r2:
                    self._corridor[key] = 0.75
                else:
                    self._corridor[key] = self._cross_region_score(r1, r2)

        self._log(f"  NumPy arrays ready (exporters={len(E)}, importers={len(I)})")

    # ── public match API ──────────────────────────────────────────────────────

    def match(self, top_k: int = 10, threshold: float = 0.0) -> pd.DataFrame:
        """
        Global best matches across all exporter−importer combinations.
        Returns the top_k pairs by Total_Score.

        ⚠️  On large datasets (12K×12K) this is slow (~minutes).
            Use match_sample() for quick iteration or when top_k is small.
        """
        self._log(f"Running global match (top_k={top_k}, threshold={threshold}) …")
        records: list[dict] = []

        for imp_idx in range(len(self.imp)):
            matches = self._score_exporter_for_importer(imp_idx, top_k=top_k)
            for m in matches:
                if m["Total_Score"] >= threshold * 100:
                    records.append(m)

        df = pd.DataFrame(records)
        if df.empty:
            return df

        df = (
            df.sort_values("Total_Score", ascending=False)
              .drop_duplicates(subset=["Exporter_ID", "Importer_Buyer_ID"])
              .head(top_k)
              .reset_index(drop=True)
        )
        self._log(f"  → {len(df)} global matches returned")
        return df

    def match_sample(
        self,
        n_importers: int = 200,
        top_k: int = 10,
        threshold: float = 0.0,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Fast sampling mode: score n_importers random importers against ALL exporters.
        Ideal for development, demos, and the Streamlit UI.

        For production / batch export, use match() with the full DataFrame.
        """
        n = min(n_importers, len(self.imp))
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(len(self.imp), size=n, replace=False).tolist()

        self._log(
            f"Running sampled match: {n} importers × {len(self.exp):,} exporters …"
        )
        records: list[dict] = []
        for imp_idx in sample_indices:
            matches = self._score_exporter_for_importer(imp_idx, top_k=top_k)
            for m in matches:
                if m["Total_Score"] >= threshold * 100:
                    records.append(m)

        df = pd.DataFrame(records)
        if df.empty:
            return df

        df = (
            df.sort_values("Total_Score", ascending=False)
              .drop_duplicates(subset=["Exporter_ID", "Importer_Buyer_ID"])
              .head(top_k)
              .reset_index(drop=True)
        )
        self._log(f"  → {len(df)} sampled matches returned")
        return df

    def match_for_exporter(
        self, exporter_id: str, top_k: int = 10, threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Return the best importer matches for a specific Exporter_ID.
        Uses vectorised NumPy scoring (same speed as match_for_importer).
        """
        exp_id_col = _col(self.exp, "exporter_id", "exp_id", "record_id", "id")
        if exp_id_col is None:
            raise ValueError("Cannot find Exporter_ID column in exporters DataFrame")

        # Try exact match first, then int-stripped version (handles '1.0' vs '1')
        search_id = str(exporter_id).strip()
        row_mask = self.exp[exp_id_col].astype(str).str.strip() == search_id
        if not row_mask.any():
            # try stripping trailing .0
            alt_id = search_id[:-2] if search_id.endswith(".0") and search_id[:-2].isdigit() else search_id
            row_mask = self.exp[exp_id_col].astype(str).str.strip() == alt_id
        if not row_mask.any():
            raise KeyError(f"Exporter '{exporter_id}' not found")

        exp_idx = int(row_mask.idxmax())

        # ── Fast vectorised path (symmetric to _score_exporter_for_importer) ──
        records = self._score_importers_for_exporter(exp_idx, top_k=top_k)
        df = (
            pd.DataFrame(records)
              .query(f"Total_Score >= {threshold * 100}")
              .reset_index(drop=True)
        )
        return df

    def match_for_importer(
        self, buyer_id: str, top_k: int = 10, threshold: float = 0.0
    ) -> pd.DataFrame:
        """Return the best exporter matches for a specific Buyer_ID."""
        imp_id_col = _col(self.imp, "buyer_id", "importer_id", "record_id", "id")
        if imp_id_col is None:
            raise ValueError("Cannot find Buyer_ID column in importers DataFrame")

        search_id = str(buyer_id).strip()
        row_mask = self.imp[imp_id_col].astype(str).str.strip() == search_id
        if not row_mask.any():
            alt_id = search_id[:-2] if search_id.endswith(".0") and search_id[:-2].isdigit() else search_id
            row_mask = self.imp[imp_id_col].astype(str).str.strip() == alt_id
        if not row_mask.any():
            raise KeyError(f"Buyer '{buyer_id}' not found")

        imp_idx = int(row_mask.idxmax())
        records = self._score_exporter_for_importer(imp_idx, top_k=top_k)
        df = (
            pd.DataFrame(records)
              .query(f"Total_Score >= {threshold * 100}")
              .reset_index(drop=True)
        )
        return df

    # ── scoring core (vectorised) ─────────────────────────────────────────────

    def _score_importers_for_exporter(
        self, exp_idx: int, top_k: int = 10
    ) -> list[dict]:
        """
        Symmetric twin of _score_exporter_for_importer.
        Score ALL importers against a single FIXED exporter using NumPy
        broadcasting. Runs in ~10 ms for 12K importers.
        """
        n_imp = len(self.imp)

        # ── P1: Product ───────────────────────────────────────────────────────
        cos_scores = cosine_similarity(
            self._exp_tfidf[exp_idx], self._imp_tfidf
        ).flatten()  # shape (n_imp,)

        exp_ind_tokens = set(re.findall(r"[a-z]{3,}", self._e_ind[exp_idx].lower()))
        if exp_ind_tokens:
            overlap = np.array([
                len(exp_ind_tokens & set(re.findall(r"[a-z]{3,}", t.lower()))) /
                max(len(exp_ind_tokens | set(re.findall(r"[a-z]{3,}", t.lower()))), 1)
                for t in self._i_ind
            ])
        else:
            overlap = np.zeros(n_imp)

        exp_cert = self._e_cert[exp_idx].lower()
        valid_exp = exp_cert not in ("", "unknown", "nan")
        valid_imp = np.array([c.lower() not in ("", "unknown", "nan") for c in self._i_cert])
        cert_match = np.array([c.lower() == exp_cert for c in self._i_cert])
        cert_scores = np.where(
            ~valid_exp | ~valid_imp, 0.2,
            np.where(cert_match, 1.0, 0.4)
        )
        p1 = np.clip(cos_scores * 0.55 + overlap * 0.30 + cert_scores * 0.15, 0, 1)

        # ── P2: Geography ─────────────────────────────────────────────────────
        exp_region = self._e_region[exp_idx]
        geo_base = np.array([
            self._corridor.get((exp_region, ir), 0.35)
            for ir in self._i_region
        ])
        state_bonus = self._e_state_bonus[exp_idx]   # scalar for this exporter
        p2 = np.clip(geo_base + state_bonus, 0, 1)

        # ── P3: High-Intent Signals ───────────────────────────────────────────
        # Exporter signal scalar
        exp_sig = float(self._e_signal_vec[exp_idx])

        # Importer composite signals (arrays)
        buyer_sig = np.clip(
            (self._i_intent + self._i_hiring + self._i_funding +
             self._i_eng + self._i_dm) / 5.0,
            0, 1,
        )
        risk_arr = np.clip(
            (self._i_war + self._i_stock + self._i_calamity +
             self._i_currency + self._i_tariff) / 5.0,
            0, 1,
        )
        news_arr = self._i_news_bonus          # shape (n_imp,)

        p3 = np.clip(
            buyer_sig * 0.50
            - risk_arr * 0.25
            + exp_sig  * 0.25
            + news_arr * 0.25,
            0, 1,
        )

        # ── P4: Trade Activity ────────────────────────────────────────────────
        exp_supply = float(self._e_supply_tons[exp_idx])
        with np.errstate(invalid='ignore', divide='ignore'):
            vol_ratio = np.where(
                (exp_supply > 0) & (self._i_avg_order > 0),
                np.minimum(self._i_avg_order, exp_supply) /
                np.maximum(self._i_avg_order, exp_supply),
                0.5,
            )

        exp_rev = float(self._e_rev[exp_idx])
        with np.errstate(invalid='ignore', divide='ignore'):
            rev_ratio = np.where(
                (exp_rev > 0) & (self._i_rev > 0),
                np.minimum(self._i_rev, exp_rev) / np.maximum(self._i_rev, exp_rev),
                0.3,
            )

        buyer_rel = np.clip(
            (self._i_good_pay + self._i_prompt + self._i_resp_prob) / 3.0, 0, 1
        )
        exp_rel = float(np.clip(
            (self._e_pay_terms[exp_idx] + self._e_resp_score[exp_idx] +
             self._e_salesnav[exp_idx]) / 3.0, 0, 1
        ))

        p4 = np.clip(
            vol_ratio * 0.35 + rev_ratio * 0.20 +
            buyer_rel * 0.25 + exp_rel   * 0.20,
            0, 1,
        )

        # ── Weighted total ────────────────────────────────────────────────────
        total = (
            p1 * PILLAR_WEIGHTS["product"]   +
            p2 * PILLAR_WEIGHTS["geography"] +
            p3 * PILLAR_WEIGHTS["signals"]   +
            p4 * PILLAR_WEIGHTS["activity"]
        ) * 100.0
        total = np.clip(total, 0, 100)

        # ── Select top_k ─────────────────────────────────────────────────────
        k = min(top_k, n_imp)
        top_indices = np.argpartition(total, -k)[-k:]
        top_indices = top_indices[np.argsort(total[top_indices])[::-1]]

        exp_id_col = _col(self.exp, "exporter_id", "exp_id", "record_id")
        imp_id_col = _col(self.imp, "buyer_id", "importer_id", "record_id")
        exp_id_val = str(self.exp.iloc[exp_idx][exp_id_col]) if exp_id_col else str(exp_idx)

        results = []
        for ii in top_indices:
            imp_id_val = str(self.imp.iloc[int(ii)][imp_id_col]) if imp_id_col else str(ii)
            results.append({
                "Exporter_ID":       exp_id_val,
                "Importer_Buyer_ID": imp_id_val,
                "Total_Score":       round(float(total[ii]), 2),
                "Grade":             _grade(float(total[ii])),
                "P1_Product":        round(float(p1[ii]) * PILLAR_WEIGHTS["product"]   * 100, 2),
                "P2_Geography":      round(float(p2[ii]) * PILLAR_WEIGHTS["geography"] * 100, 2),
                "P3_Signals":        round(float(p3[ii]) * PILLAR_WEIGHTS["signals"]   * 100, 2),
                "P4_Activity":       round(float(p4[ii]) * PILLAR_WEIGHTS["activity"]  * 100, 2),
                "Explanation":       json.dumps({
                    "1_product_compatibility": {
                        "score": round(float(p1[ii]), 4),
                        "max_pts": 30,
                        "reasons": [
                            f"TF-IDF cosine={cos_scores[ii]:.3f}",
                            f"Industry token overlap={overlap[ii]:.3f}",
                            f"Cert match={cert_scores[ii]:.2f}",
                        ],
                    },
                    "2_geography_fit": {
                        "score": round(float(p2[ii]), 4),
                        "max_pts": 20,
                        "reasons": [
                            f"Trade corridor ({exp_region} -> {self._i_region[ii]})={geo_base[ii]:.2f}",
                            f"State bonus={state_bonus:.2f}",
                        ],
                    },
                    "3_high_intent_signals": {
                        "score": round(float(p3[ii]), 4),
                        "max_pts": 30,
                        "reasons": [
                            f"Buyer signal={buyer_sig[ii]:.3f}",
                            f"Risk penalty={risk_arr[ii]:.3f}",
                            f"Exporter signal={exp_sig:.3f}",
                            f"News bonus={news_arr[ii]:.3f}",
                        ],
                    },
                    "4_trade_activity": {
                        "score": round(float(p4[ii]), 4),
                        "max_pts": 20,
                        "reasons": [
                            f"Volume fit={vol_ratio[ii]:.3f}",
                            f"Revenue compat={rev_ratio[ii]:.3f}",
                            f"Buyer reliability={buyer_rel[ii]:.3f}",
                            f"Exporter reliability={exp_rel:.3f}",
                        ],
                    },
                }),
            })
        return results

    def _score_exporter_for_importer(
        self, imp_idx: int, top_k: int = 10
    ) -> list[dict]:
        """
        Score ALL exporters against a single importer using NumPy broadcasting.
        No Python for-loop over exporters → runs in ~10 ms for 12K exporters.
        Returns the top_k dicts.
        """
        n_exp = len(self.exp)

        # ── P1: Product (vectorised) ──────────────────────────────────────────
        # 1a — TF-IDF cosine similarity (already vectorised)
        cos_scores = cosine_similarity(self._imp_tfidf[imp_idx], self._exp_tfidf).flatten()

        # 1b — Industry token Jaccard (pre-compute imp tokens once)
        imp_ind_tokens = set(re.findall(r"[a-z]{3,}", self._i_ind[imp_idx].lower()))
        if imp_ind_tokens:
            overlap = np.array([
                len(imp_ind_tokens & set(re.findall(r"[a-z]{3,}", t.lower()))) /
                max(len(imp_ind_tokens | set(re.findall(r"[a-z]{3,}", t.lower()))), 1)
                for t in self._e_ind
            ])
        else:
            overlap = np.zeros(n_exp)

        # 1c — Certification match
        imp_cert = self._i_cert[imp_idx].lower()
        valid_imp = imp_cert not in ("", "unknown", "nan")
        valid_exp = np.array([c.lower() not in ("", "unknown", "nan") for c in self._e_cert])
        cert_match = np.array([c.lower() == imp_cert for c in self._e_cert])
        cert_scores = np.where(
            ~valid_imp | ~valid_exp, 0.2,
            np.where(cert_match, 1.0, 0.4)
        )

        p1 = np.clip(cos_scores * 0.55 + overlap * 0.30 + cert_scores * 0.15, 0, 1)

        # ── P2: Geography (vectorised) ────────────────────────────────────────
        imp_region = self._i_region[imp_idx]
        geo_base = np.array([
            self._corridor.get((er, imp_region), 0.35)
            for er in self._e_region
        ])
        p2 = np.clip(geo_base + self._e_state_bonus, 0, 1)

        # ── P3: High-Intent Signals (vectorised) ─────────────────────────────
        # 3a — Importer composite signal (scalar)
        buyer_sig = float(np.mean([
            self._i_intent[imp_idx],
            self._i_hiring[imp_idx],
            self._i_funding[imp_idx],
            self._i_eng[imp_idx],
            self._i_dm[imp_idx],
        ]))

        # 3b — Risk penalty (scalar)
        risk = float(np.mean([
            self._i_war[imp_idx],
            self._i_stock[imp_idx],
            self._i_calamity[imp_idx],
            self._i_currency[imp_idx],
            self._i_tariff[imp_idx],
        ]))
        risk = min(risk, 1.0)

        # 3c — Exporter signals (array, n_exp)
        exp_sig = self._e_signal_vec  # pre-computed array

        # 3d — News bonus (scalar for this importer industry)
        news_bonus = self._i_news_bonus[imp_idx]

        p3 = np.clip(
            buyer_sig * 0.50
            - risk    * 0.25
            + exp_sig * 0.25
            + news_bonus * 0.25,
            0, 1
        )

        # ── P4: Trade Activity (vectorised) ──────────────────────────────────
        avg_order = self._i_avg_order[imp_idx]
        # Volume fit
        supply = self._e_supply_tons
        with np.errstate(invalid='ignore', divide='ignore'):
            vol_ratio = np.where(
                (supply > 0) & (avg_order > 0),
                np.minimum(avg_order, supply) / np.maximum(avg_order, supply),
                0.5           # neutral when data missing
            )

        # Revenue compatibility
        imp_rev = self._i_rev[imp_idx]
        exp_rev = self._e_rev
        with np.errstate(invalid='ignore', divide='ignore'):
            rev_ratio = np.where(
                (exp_rev > 0) & (imp_rev > 0),
                np.minimum(exp_rev, imp_rev) / np.maximum(exp_rev, imp_rev),
                0.3
            )

        # Buyer reliability (scalar)
        buyer_rel = float(np.mean([
            self._i_good_pay[imp_idx],
            self._i_prompt[imp_idx],
            self._i_resp_prob[imp_idx],
        ]))

        # Exporter reliability (array)
        exp_rel = np.clip(
            (self._e_pay_terms + self._e_resp_score + self._e_salesnav) / 3.0, 0, 1
        )

        p4 = np.clip(
            vol_ratio * 0.35 + rev_ratio * 0.20 + buyer_rel * 0.25 + exp_rel * 0.20,
            0, 1
        )

        # ── Weighted total ────────────────────────────────────────────────────
        total = (
            p1 * PILLAR_WEIGHTS["product"]   +
            p2 * PILLAR_WEIGHTS["geography"] +
            p3 * PILLAR_WEIGHTS["signals"]   +
            p4 * PILLAR_WEIGHTS["activity"]
        ) * 100.0
        total = np.clip(total, 0, 100)

        # ── Select top_k and build result dicts ───────────────────────────────
        top_indices = np.argpartition(total, -min(top_k, n_exp))[-min(top_k, n_exp):]
        top_indices = top_indices[np.argsort(total[top_indices])[::-1]]

        imp_id = str(self._i_ids[imp_idx])
        results = []
        for ei in top_indices:
            score = float(total[ei])
            explanation = {
                "1_product_compatibility": {
                    "score": round(float(p1[ei]), 4), "max_pts": 30,
                    "reasons": [
                        f"TF-IDF cosine={cos_scores[ei]:.3f} ({self._e_ind[ei]} ↔ {self._i_ind[imp_idx]})",
                        f"Industry token overlap={overlap[ei]:.3f}",
                        f"Certification: exp='{self._e_cert[ei]}' imp='{self._i_cert[imp_idx]}' → {cert_scores[ei]:.2f}",
                    ]
                },
                "2_geography_fit": {
                    "score": round(float(p2[ei]), 4), "max_pts": 20,
                    "reasons": [
                        f"Exporter region: {self._e_region[ei]} | Importer region: {imp_region}",
                        f"Corridor base={geo_base[ei]:.2f} + state_bonus={self._e_state_bonus[ei]:.2f}",
                    ]
                },
                "3_high_intent_signals": {
                    "score": round(float(p3[ei]), 4), "max_pts": 30,
                    "reasons": [
                        f"Importer signal composite={buyer_sig:.3f} (risk penalty={risk:.3f})",
                        f"Exporter live signal={self._e_signal_vec[ei]:.3f}",
                        f"News industry bonus={news_bonus:.3f}",
                    ]
                },
                "4_trade_activity": {
                    "score": round(float(p4[ei]), 4), "max_pts": 20,
                    "reasons": [
                        f"Volume fit ratio={vol_ratio[ei]:.3f} (order={avg_order:.0f}t, supply={supply[ei]:.0f}t)",
                        f"Revenue compat={rev_ratio[ei]:.3f}",
                        f"Buyer reliability={buyer_rel:.3f}",
                        f"Exporter reliability={exp_rel[ei]:.3f}",
                    ]
                },
            }
            results.append({
                "Exporter_ID":       str(self._e_ids[ei]),
                "Importer_Buyer_ID": imp_id,
                "Total_Score":       round(score, 2),
                "Grade":             _grade(score),
                "P1_Product":        round(float(p1[ei]) * 30, 2),
                "P2_Geography":      round(float(p2[ei]) * 20, 2),
                "P3_Signals":        round(float(p3[ei]) * 30, 2),
                "P4_Activity":       round(float(p4[ei]) * 20, 2),
                "Explanation":       json.dumps(explanation, ensure_ascii=False),
            })
        return results

    def _score_pair(
        self,
        exp_idx: int,
        imp_idx: int,
        precomputed_cos: float | None = None,
    ) -> dict:
        """
        Compute the full 4-pillar score for one (exporter, importer) pair.
        Returns a flat dict suitable for DataFrame construction.
        """
        exp_row = self.exp.iloc[exp_idx]
        imp_row = self.imp.iloc[imp_idx]

        # ── Pillar 1: Product Compatibility (30 pts) ─────────────────────────
        p1, p1_exp = self._score_product(exp_row, imp_row, exp_idx, imp_idx, precomputed_cos)

        # ── Pillar 2: Geography Fit (20 pts) ─────────────────────────────────
        p2, p2_exp = self._score_geography(exp_row, imp_row)

        # ── Pillar 3: High-Intent Signals (30 pts) ───────────────────────────
        p3, p3_exp = self._score_signals(exp_row, imp_row)

        # ── Pillar 4: Trade Frequency & Activity (20 pts) ────────────────────
        p4, p4_exp = self._score_activity(exp_row, imp_row)

        # ── Weighted total ────────────────────────────────────────────────────
        total = (
            p1 * PILLAR_WEIGHTS["product"]   +
            p2 * PILLAR_WEIGHTS["geography"] +
            p3 * PILLAR_WEIGHTS["signals"]   +
            p4 * PILLAR_WEIGHTS["activity"]
        ) * 100  # convert 0–1 weighted avg to 0–100

        total = round(min(max(total, 0), 100), 2)

        explanation = {
            "1_product_compatibility": p1_exp,
            "2_geography_fit":         p2_exp,
            "3_high_intent_signals":   p3_exp,
            "4_trade_activity":        p4_exp,
        }

        exp_id = _val(exp_row, "exporter_id", "exp_id", "record_id", default=f"EXP_{exp_idx}")
        imp_id = _val(imp_row, "buyer_id", "importer_id", "record_id", default=f"BYR_{imp_idx}")

        return {
            "Exporter_ID":       str(exp_id),
            "Importer_Buyer_ID": str(imp_id),
            "Total_Score":       total,
            "Grade":             _grade(total),
            "P1_Product":        round(p1 * 30, 2),
            "P2_Geography":      round(p2 * 20, 2),
            "P3_Signals":        round(p3 * 30, 2),
            "P4_Activity":       round(p4 * 20, 2),
            "Explanation":       json.dumps(explanation, ensure_ascii=False),
        }

    # ── Pillar 1: Product Compatibility ──────────────────────────────────────

    def _score_product(
        self,
        exp_row: pd.Series,
        imp_row: pd.Series,
        exp_idx: int,
        imp_idx: int,
        precomputed_cos: float | None,
    ) -> tuple[float, dict]:

        reasons: list[str] = []
        scores:  list[float] = []

        # 1a — TF-IDF cosine similarity on Industry
        cos = (
            precomputed_cos
            if precomputed_cos is not None
            else float(cosine_similarity(
                self._exp_tfidf[exp_idx], self._imp_tfidf[imp_idx]
            ).flat[0])
        )
        scores.append(cos * 0.55)  # weighted sub-contribution
        reasons.append(
            f"Industry TF-IDF cosine={cos:.3f} "
            f"(exp='{_val(exp_row, 'industry', default='?')}', "
            f"imp='{_val(imp_row, 'industry', default='?')}')"
        )

        # 1b — HS code prefix match (not in this dataset directly, but
        #       we check industry string prefix overlap as proxy)
        exp_ind = str(_val(exp_row, "industry", default="")).lower().strip()
        imp_ind = str(_val(imp_row, "industry", default="")).lower().strip()
        hs_proxy = self._industry_overlap(exp_ind, imp_ind)
        scores.append(hs_proxy * 0.30)
        reasons.append(f"Industry keyword overlap score={hs_proxy:.3f}")

        # 1c — Certification bonus (shared or compatible certs)
        exp_cert = str(_val(exp_row, "certification", default="")).lower()
        imp_cert = str(_val(imp_row, "certification", default="")).lower()
        cert_score = 0.0
        if exp_cert not in ("", "unknown", "nan") and imp_cert not in ("", "unknown", "nan"):
            if exp_cert == imp_cert:
                cert_score = 1.0
                reasons.append(f"Matching certification: '{exp_cert}'")
            else:
                cert_score = 0.4
                reasons.append(f"Different certifications (partial fit)")
        else:
            cert_score = 0.2
            reasons.append("Certification data missing")
        scores.append(cert_score * 0.15)

        total = min(sum(scores), 1.0)
        return total, {"score": round(total, 4), "max_pts": 30, "reasons": reasons}

    @staticmethod
    def _industry_overlap(a: str, b: str) -> float:
        """Token Jaccard similarity between two industry strings."""
        ta = set(re.findall(r"[a-z]{3,}", a))
        tb = set(re.findall(r"[a-z]{3,}", b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    # ── Pillar 2: Geography Fit ───────────────────────────────────────────────

    def _score_geography(
        self, exp_row: pd.Series, imp_row: pd.Series
    ) -> tuple[float, dict]:

        reasons: list[str] = []

        # Exporter geography: State (Indian context)
        exp_state   = str(_val(exp_row, "state", default="")).strip()
        exp_country = "India" if exp_state else str(_val(exp_row, "country", default="")).strip()
        imp_country = str(_val(imp_row, "country", default="Unknown")).strip()

        exp_region = _REGION_MAP.get(exp_country, _REGION_MAP.get("India", "South Asia"))
        imp_region = _REGION_MAP.get(imp_country, "Unknown")

        # Perfect country match (exporter country == importer country: rarely useful,
        # but handle the edge case)
        if exp_country and exp_country == imp_country:
            score = 1.0
            reasons.append(f"Same country: {exp_country}")
            return score, {"score": 1.0, "max_pts": 20, "reasons": reasons}

        # Macro-region match
        if imp_region != "Unknown" and exp_region == imp_region:
            base = 0.75
            reasons.append(f"Same macro-region: {exp_region}")
        elif imp_region != "Unknown":
            # Partial cross-region score based on known trade corridor proximity
            base = self._cross_region_score(exp_region, imp_region)
            reasons.append(
                f"Cross-region: {exp_region} → {imp_region} (corridor score={base:.2f})"
            )
        else:
            base = 0.3
            reasons.append(f"Importer country '{imp_country}' not in region map — default score")

        # Indian state bonus: high-export states get a small boost
        if exp_state in _INDIA_STATE_REGION:
            base = min(base + 0.05, 1.0)
            reasons.append(f"High-export Indian state: {exp_state}")

        return base, {"score": round(base, 4), "max_pts": 20, "reasons": reasons}

    @staticmethod
    def _cross_region_score(exp_region: str, imp_region: str) -> float:
        """Known strong trade corridors get higher partial scores."""
        corridors = {
            frozenset(["South Asia", "Middle East"]):    0.65,
            frozenset(["South Asia", "Southeast Asia"]): 0.60,
            frozenset(["South Asia", "East Asia"]):      0.55,
            frozenset(["South Asia", "Europe"]):         0.50,
            frozenset(["South Asia", "North America"]):  0.50,
            frozenset(["South Asia", "Africa"]):         0.55,
            frozenset(["Southeast Asia", "East Asia"]):  0.60,
            frozenset(["Europe", "North America"]):      0.55,
            frozenset(["Middle East", "Africa"]):        0.50,
        }
        return corridors.get(frozenset([exp_region, imp_region]), 0.35)

    # ── Pillar 3: High-Intent Signals ─────────────────────────────────────────

    def _score_signals(
        self, exp_row: pd.Series, imp_row: pd.Series
    ) -> tuple[float, dict]:

        reasons: list[str] = []
        sub_scores: list[float] = []

        # 3a — Importer live intent signals (direct columns)
        intent       = _safe_float(_val(imp_row, "intent_score",         default=0))
        hiring       = _safe_float(_val(imp_row, "hiring_growth",        default=0))
        funding      = _safe_float(_val(imp_row, "funding_event",        default=0))
        eng_spike    = _safe_float(_val(imp_row, "engagement_spike",     default=0))
        dm_change    = _safe_float(_val(imp_row, "decisionmaker_change", default=0))

        # Normalise intent_score (assume 0–10 scale; handle 0–1 gracefully)
        if intent > 1:
            intent = intent / 10.0
        intent = min(intent, 1.0)

        buyer_signal = np.mean([
            intent,
            min(hiring, 1.0),
            min(funding, 1.0),
            min(eng_spike, 1.0),
            min(dm_change, 1.0),
        ])
        sub_scores.append(buyer_signal * 0.50)
        reasons.append(
            f"Importer live signals — intent={intent:.2f}, hiring={hiring:.2f}, "
            f"funding={funding:.2f}, eng_spike={eng_spike:.2f}, dm_change={dm_change:.2f} "
            f"→ composite={buyer_signal:.3f}"
        )

        # 3b — Risk deductions from importer columns
        war        = _safe_float(_val(imp_row, "war_event",            default=0))
        stock_shk  = _safe_float(_val(imp_row, "stockmarket_shock",    default=0))
        calamity   = _safe_float(_val(imp_row, "natural_calamity",     default=0))
        currency   = _safe_float(_val(imp_row, "currency_fluctuation", default=0))
        tariff_news= _safe_float(_val(imp_row, "tariff_news",          default=0))

        risk_penalty = np.mean([war, stock_shk, calamity, currency, tariff_news])
        risk_penalty = min(risk_penalty, 1.0)
        sub_scores.append(-risk_penalty * 0.25)
        if risk_penalty > 0.1:
            reasons.append(
                f"Risk penalty={risk_penalty:.3f} "
                f"(war={war}, stock={stock_shk}, calamity={calamity}, "
                f"fx={currency}, tariff={tariff_news})"
            )

        # 3c — Exporter live signals
        exp_intent   = _safe_float(_val(exp_row, "intent_score",            default=0))
        exp_hiring   = _safe_float(_val(exp_row, "hiring_signal",           default=0))
        exp_linkedin = _safe_float(_val(exp_row, "linkedin_activity",       default=0))

        if exp_intent > 1:
            exp_intent = exp_intent / 10.0
        exp_intent = min(exp_intent, 1.0)

        exp_signal = np.mean([exp_intent, min(exp_hiring, 1.0), min(exp_linkedin, 1.0)])
        sub_scores.append(exp_signal * 0.25)
        reasons.append(f"Exporter signals — intent={exp_intent:.2f}, hiring={exp_hiring:.2f} → {exp_signal:.3f}")

        # 3d — Cross-reference with news index
        imp_industry = str(_val(imp_row, "industry", default="")).lower().strip()
        news_bonus = self._news_signal_bonus(imp_industry)
        sub_scores.append(news_bonus * 0.25)
        if news_bonus > 0:
            reasons.append(f"News industry signal bonus for '{imp_industry}' = {news_bonus:.3f}")
        elif news_bonus < 0:
            reasons.append(f"News industry signal PENALTY for '{imp_industry}' = {news_bonus:.3f}")

        total = min(max(sum(sub_scores), 0.0), 1.0)
        return total, {"score": round(total, 4), "max_pts": 30, "reasons": reasons}

    def _news_signal_bonus(self, industry: str) -> float:
        """
        Average sentiment signal from the news index for a given industry.
        Returns a value in [-0.5, +0.5] to act as a bonus/penalty.
        """
        if not self._news_index or not industry:
            return 0.0

        # Partial-match: any key that overlaps with the industry string
        matched_flags: list[str] = []
        for key, flags_list in self._news_index.items():
            tokens_key = set(re.findall(r"[a-z]{3,}", key))
            tokens_ind = set(re.findall(r"[a-z]{3,}", industry))
            if tokens_key & tokens_ind:
                matched_flags.extend(flags_list)

        if not matched_flags:
            return 0.0

        pos = neg = 0
        for flags_str in matched_flags:
            for flag in flags_str.split(","):
                flag = flag.strip()
                if flag in _POSITIVE_SIGNAL_FLAGS:
                    pos += 1
                elif flag in _NEGATIVE_SIGNAL_FLAGS:
                    neg += 1

        total = pos + neg
        if total == 0:
            return 0.0
        return round(((pos - neg) / total) * 0.5, 4)

    # ── Pillar 4: Trade Frequency & Activity ─────────────────────────────────

    def _score_activity(
        self, exp_row: pd.Series, imp_row: pd.Series
    ) -> tuple[float, dict]:

        reasons: list[str] = []
        sub_scores: list[float] = []

        # 4a — Order volume fit: Avg_Order_Tons (imp) vs Quantity_Tons (exp)
        avg_order   = _safe_float(_val(imp_row, "avg_order_tons",           default=0))
        exp_qty     = _safe_float(_val(exp_row, "quantity_tons",            default=0))
        cap_tons    = _safe_float(_val(exp_row, "manufacturing_capacity_tons", default=0))

        exp_supply = max(exp_qty, cap_tons)
        if exp_supply > 0 and avg_order > 0:
            ratio = min(avg_order, exp_supply) / max(avg_order, exp_supply)
            sub_scores.append(ratio * 0.35)
            reasons.append(
                f"Volume fit: imp_order={avg_order:.0f}t, exp_supply={exp_supply:.0f}t → ratio={ratio:.3f}"
            )
        else:
            sub_scores.append(0.25)
            reasons.append("Volume data missing — neutral score applied")

        # 4b — Revenue size compatibility
        exp_rev = _safe_float(_val(exp_row, "revenue_size_usd", default=0))
        imp_rev = _safe_float(_val(imp_row, "revenue_size_usd", default=0))
        if exp_rev > 0 and imp_rev > 0:
            rev_ratio = min(exp_rev, imp_rev) / max(exp_rev, imp_rev)
            sub_scores.append(rev_ratio * 0.20)
            reasons.append(f"Revenue compatibility: exp={exp_rev:.0f}, imp={imp_rev:.0f} → ratio={rev_ratio:.3f}")
        else:
            sub_scores.append(0.10)

        # 4c — Buyer reliability signals
        good_pay  = _safe_float(_val(imp_row, "good_payment_history", default=0))
        prompt    = _safe_float(_val(imp_row, "prompt_response",      default=0))
        resp_prob = _safe_float(_val(imp_row, "response_probability", default=0))

        if resp_prob > 1:
            resp_prob = resp_prob / 100.0   # handle percentage format

        buyer_rel = np.mean([
            min(good_pay, 1.0),
            min(prompt,   1.0),
            min(resp_prob, 1.0),
        ])
        sub_scores.append(buyer_rel * 0.25)
        reasons.append(
            f"Buyer reliability — payment={good_pay:.2f}, response={resp_prob:.2f} → {buyer_rel:.3f}"
        )

        # 4d — Exporter reliability signals
        exp_payment = _safe_float(_val(exp_row, "good_payment_terms",    default=0))
        exp_resp    = _safe_float(_val(exp_row, "prompt_response_score", default=0))
        salesnav    = _safe_float(_val(exp_row, "salesnav_profileviews", default=0))

        if salesnav > 1:
            salesnav = min(salesnav / 1000.0, 1.0)  # normalise large view counts

        exp_rel = np.mean([min(exp_payment, 1.0), min(exp_resp, 1.0), min(salesnav, 1.0)])
        sub_scores.append(exp_rel * 0.20)
        reasons.append(f"Exporter reliability — payment={exp_payment:.2f}, resp={exp_resp:.2f} → {exp_rel:.3f}")

        total = min(max(sum(sub_scores), 0.0), 1.0)
        return total, {"score": round(total, 4), "max_pts": 20, "reasons": reasons}

    # ── utils ─────────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_matching(
    excel_path: str = "data/EXIM_DatasetAlgo_Hackathon.xlsx",
    top_k: int = 20,
    threshold: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    One-liner: load → process → match → return results DataFrame.

    Example
    -------
    >>> from models.matchmaker import run_matching
    >>> df = run_matching(top_k=10)
    >>> print(df[["Exporter_ID", "Importer_Buyer_ID", "Total_Score", "Grade"]].head(10))
    """
    from utils.data_processor import load_and_process

    data   = load_and_process(excel_path, verbose=verbose)
    engine = MatchmakingEngine(
        exporters_df=data.exporters,
        importers_df=data.importers,
        news_signals_df=data.news_signals,
        verbose=verbose,
    )
    return engine.match(top_k=top_k, threshold=threshold)


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import io
    import sys
    from pathlib import Path

    # Ensure project root is on sys.path when running as a script
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    # Fix Windows console encoding (cp1252 → utf-8)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    excel = sys.argv[1] if len(sys.argv) > 1 else "data/EXIM_DatasetAlgo_Hackathon.xlsx"
    print("\n" + "=" * 65)
    print("  TradeMatch LOC -- MatchmakingEngine Standalone Test")
    print("=" * 65 + "\n")

    from utils.data_processor import load_and_process
    data = load_and_process(excel, verbose=True)
    engine = MatchmakingEngine(
        exporters_df=data.exporters,
        importers_df=data.importers,
        news_signals_df=data.news_signals,
        verbose=True,
    )
    results = engine.match_sample(n_importers=200, top_k=15)

    if results.empty:
        print("No matches generated. Check your data.")
        sys.exit(1)

    display_cols = [
        "Exporter_ID", "Importer_Buyer_ID", "Total_Score", "Grade",
        "P1_Product", "P2_Geography", "P3_Signals", "P4_Activity",
    ]
    print("\n-- Top 15 Matches --\n")
    print(results[display_cols].to_string(index=False))

    print("\n-- Explanation (top match) --\n")
    top = results.iloc[0]
    exp_dict = json.loads(top["Explanation"])
    for pillar, detail in exp_dict.items():
        print(f"\n  {pillar.upper()} | score={detail['score']} / {detail['max_pts']} pts")
        for reason in detail["reasons"]:
            # Replace non-ASCII chars for safe Windows console display
            safe = reason.replace("\u2194", "<->").replace("\u2192", "->")
            print(f"    - {safe}")

    out_path = "data/match_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\n[OK] Results saved to {out_path}")

