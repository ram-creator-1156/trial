"""
utils/live_signals.py
═════════════════════
Real-Time Adaptability module — satisfies the hackathon "live signal" requirement.

Public API
──────────
  simulate_live_signals(engine, n_buyers, seed, scenario)
      → LiveSignalResult

The function:
  1. Randomly selects `n_buyers` importers from the engine's dataset.
  2. Injects a realistic "signal event" (bulk-buy announcement, funding round,
     tariff cut, geopolitical risk, LinkedIn hiring spike, etc.).
  3. Mutates ONLY the engine's internal NumPy arrays — no data reload needed.
  4. Re-runs match_sample() with the boosted/penalised importers to produce
     updated match rankings.
  5. Computes a before-vs-after delta so the UI can highlight rank changes.
  6. Returns a LiveSignalResult dataclass with everything the UI needs.

Usage (Streamlit button)
────────────────────────
  from utils.live_signals import simulate_live_signals, SCENARIO_NAMES
  result = simulate_live_signals(engine, n_buyers=5, scenario="bulk_buy")
  st.dataframe(result.delta_df)
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each scenario defines:
#   description : human-readable blurb for the UI card
#   boosts      : dict of engine importer array name → multiplier (>1 = boost, <1 = penalty)
#   additive    : dict of engine importer array name → flat value to add (clamped 0–1 after)
#   news_bonus  : direct additive to _i_news_bonus for affected importers
#   icon        : emoji for the UI

SCENARIOS: dict[str, dict] = {
    "bulk_buy": {
        "icon": "📦",
        "description": (
            "Breaking news: selected importers announce a bulk-purchase programme "
            "— massive volume demand surge detected on trade portals."
        ),
        "additive": {
            "_i_intent":   +0.45,
            "_i_funding":  +0.30,
            "_i_eng":      +0.35,
            "_i_hiring":   +0.20,
        },
        "boosts": {
            "_i_avg_order": 2.0,         # double the average order tonnage
        },
        "news_bonus": +0.35,
        "risk_delta": 0.0,
    },
    "funding_round": {
        "icon": "💰",
        "description": (
            "Series-B funding confirmed for selected importers — "
            "expansion into new geographies imminent."
        ),
        "additive": {
            "_i_funding":  +0.60,
            "_i_hiring":   +0.40,
            "_i_dm":       +0.30,   # decision-maker change likely
            "_i_eng":      +0.25,
        },
        "boosts": {
            "_i_rev":      1.5,          # revenue signal boost
        },
        "news_bonus": +0.25,
        "risk_delta": 0.0,
    },
    "tariff_cut": {
        "icon": "✂️",
        "description": (
            "Government announces 15% tariff reduction on selected industry — "
            "import cost drops making deal economics highly attractive."
        ),
        "additive": {
            "_i_intent":  +0.35,
            "_i_tariff":  -0.50,    # tariff_news risk goes DOWN (positive)
        },
        "boosts": {},
        "news_bonus": +0.40,
        "risk_delta": -0.15,        # overall risk reduction
    },
    "linkedin_spike": {
        "icon": "📈",
        "description": (
            "Hiring spike + LinkedIn activity surge detected for selected importers — "
            "a strong leading indicator of imminent import orders."
        ),
        "additive": {
            "_i_hiring":   +0.55,
            "_i_eng":      +0.45,
            "_i_dm":       +0.20,
            "_i_intent":   +0.25,
        },
        "boosts": {},
        "news_bonus": +0.15,
        "risk_delta": 0.0,
    },
    "geopolitical_risk": {
        "icon": "⚠️",
        "description": (
            "Geopolitical tension escalates in selected importer regions — "
            "risk flags elevated; scores penalised accordingly."
        ),
        "additive": {
            "_i_war":      +0.60,
            "_i_stock":    +0.30,
            "_i_currency": +0.40,
            "_i_intent":   -0.30,
        },
        "boosts": {},
        "news_bonus": -0.40,
        "risk_delta": +0.35,
    },
    "new_trade_deal": {
        "icon": "🤝",
        "description": (
            "New bilateral trade agreement signed — "
            "selected importer markets open up with preferential access for Indian exporters."
        ),
        "additive": {
            "_i_intent":   +0.40,
            "_i_funding":  +0.20,
            "_i_tariff":   -0.30,
        },
        "boosts": {
            "_i_avg_order": 1.4,
        },
        "news_bonus": +0.45,
        "risk_delta": -0.20,
    },
    "supply_shock": {
        "icon": "🔴",
        "description": (
            "Supply chain disruption in selected regions — "
            "importers urgently seeking alternative suppliers, intent surges."
        ),
        "additive": {
            "_i_intent":   +0.55,
            "_i_eng":      +0.40,
            "_i_dm":       +0.35,
            "_i_calamity": +0.20,    # natural calamity risk component
        },
        "boosts": {
            "_i_avg_order": 1.8,
        },
        "news_bonus": +0.20,
        "risk_delta": +0.10,
    },
}

# Friendly display names for the UI dropdown
SCENARIO_NAMES: dict[str, str] = {
    "bulk_buy":          "📦  Bulk Purchase Announcement",
    "funding_round":     "💰  Series-B Funding Round",
    "tariff_cut":        "✂️   Government Tariff Cut",
    "linkedin_spike":    "📈  LinkedIn / Hiring Spike",
    "geopolitical_risk": "⚠️   Geopolitical Risk Event",
    "new_trade_deal":    "🤝  New Bilateral Trade Deal",
    "supply_shock":      "🔴  Supply Chain Shock",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiveSignalResult:
    """Everything the Streamlit UI needs to display the simulation result."""

    scenario:          str
    scenario_icon:     str
    scenario_desc:     str
    affected_buyers:   list[str]            # Buyer_IDs that received the signal
    n_affected:        int

    before_df:         pd.DataFrame         # top matches before simulation
    after_df:          pd.DataFrame         # top matches after simulation
    delta_df:          pd.DataFrame         # merged with rank_change column

    timestamp:         str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    # Per-signal event log (for UI "live feed" panel)
    event_log:         list[dict] = field(default_factory=list)

    def top_movers(self, n: int = 5) -> pd.DataFrame:
        """Return the n pairs with the largest positive rank improvement."""
        if "Rank_Change" not in self.delta_df.columns:
            return self.delta_df.head(n)
        return (
            self.delta_df
            .sort_values("Rank_Change", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def top_losers(self, n: int = 5) -> pd.DataFrame:
        """Return the n pairs with the largest negative rank change."""
        if "Rank_Change" not in self.delta_df.columns:
            return self.delta_df.tail(n)
        return (
            self.delta_df
            .sort_values("Rank_Change", ascending=True)
            .head(n)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Core simulation function
# ─────────────────────────────────────────────────────────────────────────────

def simulate_live_signals(
    engine,                          # MatchmakingEngine (modified in-place, then restored)
    n_buyers: int = 5,               # how many importers receive the signal
    seed: int | None = None,         # reproducibility; None = random
    scenario: str = "bulk_buy",      # key from SCENARIOS dict
    top_k: int = 20,                 # matches to show in before/after
    n_sample_importers: int = 300,   # importers pool used for match_sample
) -> LiveSignalResult:
    """
    Simulate a live market signal event and show how match rankings change.

    Parameters
    ----------
    engine              : An initialised MatchmakingEngine.
    n_buyers            : Number of importers to inject signals into.
    seed                : Random seed (pass the same seed to reproduce results).
    scenario            : One of the keys in SCENARIOS (see SCENARIO_NAMES for UI labels).
    top_k               : How many top matches to include in before/after DataFrames.
    n_sample_importers  : Size of the importer sample for match_sample() calls.

    Returns
    -------
    LiveSignalResult — a dataclass with before_df, after_df, delta_df, event_log, etc.
    """
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Valid options: {list(SCENARIOS.keys())}"
        )

    rng = np.random.default_rng(seed if seed is not None else random.randint(0, 99999))
    cfg = SCENARIOS[scenario]

    n_imp = len(engine.imp)
    n_buyers = min(n_buyers, n_imp)

    # ── Step 1: Run BEFORE match ──────────────────────────────────────────────
    before_df = engine.match_sample(
        n_importers=n_sample_importers,
        top_k=top_k,
        random_state=int(rng.integers(0, 999999)),
    ).copy()
    before_df["Before_Rank"] = range(1, len(before_df) + 1)

    # ── Step 2: Choose affected buyers ───────────────────────────────────────
    affected_indices = rng.choice(n_imp, size=n_buyers, replace=False).tolist()
    affected_ids     = [str(engine._i_ids[i]) for i in affected_indices]

    # ── Step 3: Snapshot existing arrays (so we can restore them) ────────────
    _MUTABLE_ARRAYS = [
        "_i_intent", "_i_hiring", "_i_funding", "_i_eng", "_i_dm",
        "_i_war", "_i_stock", "_i_calamity", "_i_currency", "_i_tariff",
        "_i_avg_order", "_i_rev", "_i_good_pay", "_i_prompt", "_i_resp_prob",
        "_i_news_bonus",
    ]
    snapshots: dict[str, np.ndarray] = {
        arr: getattr(engine, arr).copy()
        for arr in _MUTABLE_ARRAYS
        if hasattr(engine, arr)
    }

    # ── Step 4: Apply signal boosts/penalties to affected importers ───────────
    event_log: list[dict] = []
    for imp_idx in affected_indices:
        buyer_id = str(engine._i_ids[imp_idx])

        # Additive boosts (clamp to [0, 1] for probability-style columns)
        for arr_name, delta in cfg.get("additive", {}).items():
            if hasattr(engine, arr_name):
                arr = getattr(engine, arr_name)
                old_val = float(arr[imp_idx])
                arr[imp_idx] = float(np.clip(arr[imp_idx] + delta, 0.0, 1.0))
                event_log.append({
                    "buyer_id": buyer_id,
                    "signal":   arr_name.replace("_i_", ""),
                    "before":   round(old_val, 4),
                    "after":    round(float(arr[imp_idx]), 4),
                    "delta":    round(delta, 4),
                })

        # Multiplicative boosts (no upper clamp — volume/revenue can grow freely)
        for arr_name, mult in cfg.get("boosts", {}).items():
            if hasattr(engine, arr_name):
                arr = getattr(engine, arr_name)
                old_val = float(arr[imp_idx])
                arr[imp_idx] = float(arr[imp_idx] * mult)
                event_log.append({
                    "buyer_id": buyer_id,
                    "signal":   arr_name.replace("_i_", "") + "_x" + str(mult),
                    "before":   round(old_val, 4),
                    "after":    round(float(arr[imp_idx]), 4),
                    "delta":    round(float(arr[imp_idx]) - old_val, 4),
                })

        # News bonus (direct adjustment to the pre-computed bonus array)
        news_bonus_delta = cfg.get("news_bonus", 0.0)
        if news_bonus_delta and hasattr(engine, "_i_news_bonus"):
            old_val = float(engine._i_news_bonus[imp_idx])
            engine._i_news_bonus[imp_idx] = float(
                np.clip(engine._i_news_bonus[imp_idx] + news_bonus_delta, -0.5, 0.5)
            )
            event_log.append({
                "buyer_id": buyer_id,
                "signal":   "news_bonus",
                "before":   round(old_val, 4),
                "after":    round(float(engine._i_news_bonus[imp_idx]), 4),
                "delta":    round(news_bonus_delta, 4),
            })

    # ── Step 5: Run AFTER match with boosted signals ──────────────────────────
    after_df = engine.match_sample(
        n_importers=n_sample_importers,
        top_k=top_k,
        random_state=int(rng.integers(0, 999999)),
    ).copy()
    after_df["After_Rank"] = range(1, len(after_df) + 1)

    # ── Step 6: Restore original arrays (engine left clean for next call) ─────
    for arr_name, snapshot in snapshots.items():
        setattr(engine, arr_name, snapshot)

    # ── Step 7: Build delta DataFrame ────────────────────────────────────────
    delta_df = _build_delta(before_df, after_df, affected_ids)

    # ── Step 8: Assemble and return result ────────────────────────────────────
    return LiveSignalResult(
        scenario=scenario,
        scenario_icon=cfg["icon"],
        scenario_desc=cfg["description"],
        affected_buyers=affected_ids,
        n_affected=n_buyers,
        before_df=before_df.drop(columns=["Before_Rank"], errors="ignore"),
        after_df=after_df.drop(columns=["After_Rank"], errors="ignore"),
        delta_df=delta_df,
        event_log=event_log,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: build the before-vs-after delta table
# ─────────────────────────────────────────────────────────────────────────────

def _build_delta(
    before_df: pd.DataFrame,
    after_df:  pd.DataFrame,
    affected_ids: list[str],
) -> pd.DataFrame:
    """
    Merge before and after match DataFrames on (Exporter_ID, Importer_Buyer_ID),
    compute rank changes, and flag affected importers.
    """
    key   = ["Exporter_ID", "Importer_Buyer_ID"]
    score = "Total_Score"

    b = before_df[key + [score, "Grade", "Before_Rank"]].copy()
    a = after_df[key + [score, "After_Rank"]].copy()
    a.rename(columns={score: "Score_After"}, inplace=True)
    b.rename(columns={score: "Score_Before"}, inplace=True)

    merged = pd.merge(b, a, on=key, how="outer")
    merged["Before_Rank"]  = merged["Before_Rank"].fillna(99).astype(int)
    merged["After_Rank"]   = merged["After_Rank"].fillna(99).astype(int)
    merged["Score_Before"] = merged["Score_Before"].fillna(0.0)
    merged["Score_After"]  = merged["Score_After"].fillna(0.0)
    merged["Score_Delta"]  = (merged["Score_After"] - merged["Score_Before"]).round(2)

    # Positive rank_change = moved UP in rankings
    merged["Rank_Change"]  = merged["Before_Rank"] - merged["After_Rank"]

    # Flag affected importers
    merged["Signal_Injected"] = merged["Importer_Buyer_ID"].isin(affected_ids)

    # Visual indicator for UI
    merged["Movement"] = merged["Rank_Change"].apply(lambda x:
        "⬆ Up"   if x > 0 else
        "⬇ Down" if x < 0 else
        "— Flat"
    )

    return (
        merged
        .sort_values("After_Rank")
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit-ready display helper
# ─────────────────────────────────────────────────────────────────────────────

def render_simulation_result(result: LiveSignalResult) -> None:
    """
    Convenience function: when called inside a Streamlit app this renders the
    full simulation result in a well-formatted layout.

    Import this in frontend/app.py:
        from utils.live_signals import render_simulation_result
    """
    try:
        import streamlit as st
    except ImportError:
        raise ImportError("render_simulation_result() requires Streamlit to be installed.")

    # Header card
    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
                    border:1px solid rgba(255,255,255,0.1);border-radius:14px;
                    padding:20px;margin-bottom:16px;'>
          <h3 style='color:#00d2ff;margin:0'>{result.scenario_icon} Live Signal Event Fired!</h3>
          <p style='color:#ccc;margin:8px 0 4px'>{result.scenario_desc}</p>
          <small style='color:#888'>
            🕐 {result.timestamp} &nbsp;|&nbsp;
            🎯 {result.n_affected} importers affected &nbsp;|&nbsp;
            🆔 {', '.join(result.affected_buyers[:3])}{'…' if len(result.affected_buyers) > 3 else ''}
          </small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Importers Boosted", result.n_affected)
    gainers = (result.delta_df["Rank_Change"] > 0).sum()
    losers  = (result.delta_df["Rank_Change"] < 0).sum()
    col2.metric("Pairs Moved Up ⬆", int(gainers))
    col3.metric("Pairs Moved Down ⬇", int(losers))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Rankings Delta", "⬆ Top Movers", "⬇ Top Losers", "📡 Signal Event Log"]
    )

    with tab1:
        st.caption("Merged before-vs-after rankings. Green = signal-injected importer.")
        show_cols = [
            "Exporter_ID", "Importer_Buyer_ID", "Score_Before", "Score_After",
            "Score_Delta", "Before_Rank", "After_Rank", "Rank_Change",
            "Movement", "Signal_Injected",
        ]
        avail = [c for c in show_cols if c in result.delta_df.columns]
        st.dataframe(
            result.delta_df[avail].style.apply(
                lambda row: [
                    "background-color:#1a3a1a" if row.get("Signal_Injected") else ""
                    for _ in row
                ],
                axis=1,
            ),
            use_container_width=True,
        )

    with tab2:
        st.caption(f"Top 5 pairs that gained rank after the signal")
        st.dataframe(result.top_movers(5), use_container_width=True)

    with tab3:
        st.caption(f"Top 5 pairs that lost rank after the signal")
        st.dataframe(result.top_losers(5), use_container_width=True)

    with tab4:
        st.caption("Raw signal mutations applied per buyer")
        if result.event_log:
            log_df = pd.DataFrame(result.event_log)
            st.dataframe(log_df, use_container_width=True)

            # Mini live-feed style display
            st.markdown("**📡 Signal Feed**")
            for ev in result.event_log[:20]:
                delta_str = f"+{ev['delta']:.3f}" if ev["delta"] >= 0 else f"{ev['delta']:.3f}"
                colour    = "#00c853" if ev["delta"] >= 0 else "#ff5252"
                st.markdown(
                    f"<span style='color:#aaa;font-size:0.82em'>"
                    f"[{ev['buyer_id']}] <b style='color:#00d2ff'>{ev['signal']}</b>"
                    f" : {ev['before']:.3f} → "
                    f"<b style='color:{colour}'>{ev['after']:.3f}</b>"
                    f" ({delta_str})</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No signal events recorded.")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from utils.data_processor import load_and_process
    from models.matchmaker import MatchmakingEngine

    excel = sys.argv[1] if len(sys.argv) > 1 else "data/EXIM_DatasetAlgo_Hackathon.xlsx"
    print("\nLoading data...")
    data = load_and_process(excel, verbose=False)
    engine = MatchmakingEngine(
        data.exporters, data.importers, data.news_signals, verbose=True
    )

    for scenario_key in list(SCENARIOS.keys())[:3]:
        print(f"\n{'='*60}")
        print(f"Running scenario: {SCENARIO_NAMES[scenario_key]}")
        print("=" * 60)
        result = simulate_live_signals(
            engine, n_buyers=4, seed=42, scenario=scenario_key, top_k=10
        )
        print(f"Affected buyers : {result.affected_buyers}")
        print(f"Signal events   : {len(result.event_log)}")
        print(f"\nTop movers (rank improved):")
        print(result.top_movers(3)[
            ["Exporter_ID", "Importer_Buyer_ID", "Score_Before", "Score_After",
             "Rank_Change", "Signal_Injected"]
        ].to_string(index=False))

    print("\n[OK] Simulation test complete.")
