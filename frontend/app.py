"""
frontend/app.py
═══════════════
TradeMatch LOC — Swipe-to-Export  |  Streamlit Swipe UI
Run: streamlit run frontend/app.py

Dependencies
────────────
  pip install streamlit streamlit-swipecards plotly requests
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
API = "http://localhost:8000"

st.set_page_config(
    page_title="TradeMatch LOC · Swipe to Export",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal dark CSS (no neon) ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    /* Card shell */
    .match-card {
        background: linear-gradient(160deg, #1a1f2e 0%, #141826 100%);
        border: 1px solid #2a2f3e;
        border-radius: 18px;
        padding: 28px 32px;
        box-shadow: 0 8px 40px rgba(0,0,0,.45);
    }
    /* Grade pill */
    .grade { display:inline-block; padding:3px 14px; border-radius:20px;
             font-weight:700; font-size:.85em; border:1px solid; }
    /* Reason chip */
    .reason-chip { display:inline-block; background:rgba(255,255,255,.05);
                   border:1px solid rgba(255,255,255,.1); border-radius:6px;
                   padding:3px 10px; margin:2px; font-size:.78em; color:#aaa; }
    /* Outreach box */
    .email-box { background:#0d111c; border:1px solid #2a2f3e; border-radius:12px;
                 padding:18px 22px; font-size:.9em; line-height:1.8;
                 white-space:pre-wrap; color:#d0d4e0; }
    /* Section label */
    .slabel { font-size:.7em; letter-spacing:1.4px; text-transform:uppercase;
              color:#4d5468; margin-bottom:4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "card_skip":       0,        # how many top cards have been swiped past
    "exp_id":          None,
    "match":           None,
    "outreach":        None,
    "show_outreach":   False,
    "tone":            "professional",
    "passed":          [],
    "connected":       [],
    "sim_ran":         False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
#  API helpers
# ─────────────────────────────────────────────────────────────────────────────
def _api_ok() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=3).status_code == 200
    except Exception:
        return False


def _fetch_exporters() -> list[str]:
    try:
        r = requests.get(f"{API}/api/exporters", params={"limit": 200}, timeout=5)
        return r.json().get("exporter_ids", [])
    except Exception:
        return []


def _fetch_next_match(exp_id: str, skip: int = 0) -> dict | None:
    """Call /api/next_match/{exp_id}?skip=N — our new dedicated endpoint."""
    try:
        r = requests.get(
            f"{API}/api/next_match/{exp_id}",
            params={"skip": skip},
            timeout=120,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json().get("match")
    except Exception as e:
        st.error(f"Could not fetch match: {e}")
        return None


def _fetch_outreach(exp_id: str, match: dict, tone: str) -> dict | None:
    payload = {
        "exporter": {
            "exporter_id":  exp_id,
            "company_name": exp_id,
            "industry":     "General Trade",
            "state":        "India",
            "certifications": None,
            "capacity_tons":  None,
            "revenue_usd":    None,
        },
        "importer": {
            "buyer_id":     match.get("importer_buyer_id", ""),
            "company_name": match.get("importer_buyer_id", ""),
            "country":      match.get("country", "International"),
            "industry":     match.get("industry", "General Trade"),
            "avg_order_tons":       match.get("avg_order_tons"),
            "revenue_usd":          match.get("revenue_usd"),
            "response_probability": match.get("response_probability"),
        },
        "match_score":  match.get("total_score"),
        "explanation":  match.get("explainability_dict", {}),
        "tone":         tone,
        "use_llm":      False,
    }
    try:
        r = requests.post(f"{API}/api/generate_outreach", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _simulate_shift(scenario: str, n_buyers: int) -> dict | None:
    try:
        r = requests.post(
            f"{API}/api/simulate_real_time",
            json={"scenario": scenario, "n_buyers": n_buyers, "top_k": 10, "seed": 0},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────
PILLAR_COLOUR = {
    "product_compatibility": "#c9a84c",
    "geography_fit":         "#7b9cc4",
    "high_intent_signals":   "#6aaa84",
    "trade_activity":        "#b07b6a",
}

def _radar_chart(exp_dict: dict) -> go.Figure:
    """Radar chart built from the explainability_dict."""
    labels = [v["label"] for v in exp_dict.values()]
    pcts   = [v["pct"]   for v in exp_dict.values()]
    cats   = labels + [labels[0]]
    vals   = pcts   + [pcts[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats,
        fill="toself",
        fillcolor="rgba(201,168,76,.12)",
        line=dict(color="#c9a84c", width=2),
        marker=dict(size=6, color="#c9a84c"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="rgba(255,255,255,.07)",
                tickfont=dict(color="#4d5468", size=9),
                ticksuffix="%",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,.07)",
                tickfont=dict(color="#8990a8", size=11),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=30, b=30),
        showlegend=False,
        height=280,
    )
    return fig


def _grade_color(g: str) -> str:
    return {
        "A+": "#c9a84c", "A": "#8aaa6a", "B+": "#788daa",
        "B":  "#8a7aaa", "C": "#aa7a6a", "D": "#7a5a5a",
    }.get(g, "#888")

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center;padding:8px 0 14px'>
          <span style='font-size:2rem'>🌐</span>
          <h2 style='margin:4px 0 0;color:#c9a84c;font-size:1.1rem'>TradeMatch LOC</h2>
          <p style='color:#4d5468;font-size:.75rem;margin:2px 0 0'>Swipe to Export · AI Matchmaking</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_live = _api_ok()
    if api_live:
        st.success("🟢 Backend Online")
    else:
        st.error("🔴 Backend Offline")
        st.caption("Start: `uvicorn backend.main:app --reload`")

    st.divider()

    # ── Exporter picker ────────────────────────────────────────────────────
    st.markdown("### 🏭 Select Exporter")
    if api_live:
        exp_ids = _fetch_exporters()
    else:
        exp_ids = []

    selected_exp = st.selectbox(
        "Exporter ID",
        options=exp_ids or [""],
        index=0,
        key="exp_picker",
        placeholder="Choose an exporter…",
    )

    if st.button("🔍 Load Matches", type="primary", use_container_width=True):
        st.session_state["exp_id"]        = selected_exp
        st.session_state["card_skip"]     = 0
        st.session_state["show_outreach"] = False
        st.session_state["outreach"]      = None
        st.session_state["passed"]        = []
        with st.spinner("Scoring importers…"):
            m = _fetch_next_match(selected_exp, skip=0)
        st.session_state["match"] = m
        if m:
            st.toast(f"Match loaded for {selected_exp}!", icon="🎯")

    st.divider()

    # ── Live Market Shift ──────────────────────────────────────────────────
    st.markdown("### ⚡ Simulate Live Market Shift")
    SCENARIOS = {
        "bulk_buy":          "📦 Bulk Purchase Announcement",
        "funding_round":     "💰 Series-B Funding Round",
        "tariff_cut":        "✂️ Government Tariff Cut",
        "linkedin_spike":    "📈 LinkedIn / Hiring Spike",
        "geopolitical_risk": "⚠️ Geopolitical Risk Event",
        "new_trade_deal":    "🤝 New Bilateral Trade Deal",
        "supply_shock":      "🔴 Supply Chain Shock",
    }
    scenario  = st.selectbox("Signal Event", options=list(SCENARIOS.keys()),
                             format_func=lambda k: SCENARIOS[k])
    n_buyers  = st.slider("Buyers affected", 1, 15, 5)

    if st.button("⚡ Fire Signal", use_container_width=True):
        if not api_live:
            st.error("Backend must be running.")
        else:
            with st.spinner("Injecting live signal…"):
                result = _simulate_shift(scenario, n_buyers)
            if result and "error" not in result:
                up   = result.get("pairs_moved_up",   0)
                down = result.get("pairs_moved_down",  0)
                icon = result.get("scenario_icon", "⚡")
                st.toast(f"{icon} Signal fired!  ↑{up} pairs  ↓{down} pairs", icon="🔴")
                st.session_state["sim_ran"] = True
                # Refresh match for current exporter
                if st.session_state.get("exp_id"):
                    m = _fetch_next_match(
                        st.session_state["exp_id"],
                        skip=st.session_state["card_skip"],
                    )
                    st.session_state["match"] = m
            else:
                st.error(f"Error: {(result or {}).get('error', 'unknown')}")

    st.divider()

    # ── Stats ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Connected", len(st.session_state.get("connected", [])))
    c2.metric("⏭ Passed",   len(st.session_state.get("passed",    [])))

    # ── Connected list ──────────────────────────────────────────────────────
    conn = st.session_state.get("connected", [])
    if conn:
        st.divider()
        st.markdown("### ⭐ Connected")
        for c in conn:
            st.markdown(
                f"<div class='reason-chip'>🛒 {c['id']}"
                f"  <b style='color:#c9a84c'>{c['score']:.0f}pts</b></div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
#  Main screen
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='padding:6px 0 18px'>
      <h1 style='font-size:2rem;margin:0;color:#c9a84c'>🌐 Swipe to Export</h1>
      <p style='color:#4d5468;margin:4px 0 0;font-size:.9em'>
        AI-powered trade matchmaking · Explainable scoring · Live signal adaptation
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

match   = st.session_state.get("match")
exp_id  = st.session_state.get("exp_id")

# ── Empty state ───────────────────────────────────────────────────────────────
if not match or not exp_id:
    st.markdown(
        """
        <div style='text-align:center;padding:80px 40px;background:rgba(255,255,255,.01);
                    border:1px dashed #2a2f3e;border-radius:18px;margin-top:20px'>
          <div style='font-size:3.5rem;margin-bottom:14px'>🌐</div>
          <h3 style='color:#4d5468;font-weight:400'>No match loaded yet</h3>
          <p style='color:#2a2f3e'>Select an exporter and click <b style='color:#c9a84c'>Load Matches</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  Match Card
# ─────────────────────────────────────────────────────────────────────────────
imp_id     = match.get("importer_buyer_id", "—")
score      = match.get("total_score", 0)
grade      = match.get("grade", "—")
gc         = _grade_color(grade)
exp_dict   = match.get("explainability_dict", {})

skip       = st.session_state.get("card_skip", 0)

# Card header
st.markdown(
    f"""
    <div class='match-card'>
      <div style='display:flex;justify-content:space-between;align-items:flex-start'>
        <div>
          <p class='slabel'>Best Match  ·  Card {skip + 1}</p>
          <h2 style='margin:0 0 6px;font-size:1.6rem;color:#e2e5ef'>🛒 {imp_id}</h2>
          <span class='grade' style='color:{gc};border-color:{gc}55;background:{gc}18'>
            ✦ Grade {grade}
          </span>
        </div>
        <div style='text-align:center;width:90px;height:90px;border-radius:50%;
                    border:3px solid {gc};display:flex;flex-direction:column;
                    align-items:center;justify-content:center'>
          <span style='font-size:1.9rem;font-weight:700;color:{gc}'>{score:.0f}</span>
          <span style='font-size:.6rem;color:#4d5468'>/ 100</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ── Two columns: Radar + Pillar metrics ───────────────────────────────────────
col_radar, col_metrics = st.columns([1, 1], gap="large")

with col_radar:
    st.markdown("<p class='slabel'>Compatibility Radar</p>", unsafe_allow_html=True)
    if exp_dict:
        st.plotly_chart(_radar_chart(exp_dict), use_container_width=True, key="radar_main")

with col_metrics:
    st.markdown("<p class='slabel'>Score Breakdown</p>", unsafe_allow_html=True)
    COLOURS = {"product_compatibility":"#c9a84c","geography_fit":"#7b9cc4",
               "high_intent_signals":"#6aaa84","trade_activity":"#b07b6a"}
    for key, pillar in exp_dict.items():
        pct  = pillar.get("pct", 0)
        col  = COLOURS.get(key, "#888")
        pts  = pillar.get("score", 0)
        maxp = pillar.get("max_pts", 0)
        st.markdown(
            f"""
            <div style='margin-bottom:10px'>
              <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                <span style='font-size:.8em;color:#8990a8'>{pillar['label']}</span>
                <span style='font-size:.78em;font-family:monospace;color:#4d5468'>{pts:.1f}/{maxp}</span>
              </div>
              <div style='height:5px;background:#2a2f3e;border-radius:3px;overflow:hidden'>
                <div style='width:{pct}%;height:100%;background:{col};border-radius:3px;
                            transition:width .5s'></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Explainability accordion ──────────────────────────────────────────────────
with st.expander("🔍 Why this match? (Explainable AI)", expanded=False):
    for key, pillar in exp_dict.items():
        reasons = pillar.get("reasons", [])
        if reasons:
            st.markdown(f"**{pillar['label']}**")
            for r in reasons:
                safe_r = r.replace("<->", "↔").replace("->", "→")
                st.markdown(f"- {safe_r}")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Swipe Buttons  (Pass / Connect)
# ─────────────────────────────────────────────────────────────────────────────
btn_l, btn_r, _ = st.columns([1, 1, 2])

with btn_l:
    if st.button("👈  Pass  (Skip)", use_container_width=True, key="pass_btn"):
        st.session_state["passed"].append(imp_id)
        st.session_state["card_skip"] += 1
        st.session_state["show_outreach"] = False
        st.session_state["outreach"]      = None
        with st.spinner("Loading next match…"):
            nm = _fetch_next_match(exp_id, skip=st.session_state["card_skip"])
        st.session_state["match"] = nm
        st.toast(f"Passed on {imp_id}", icon="⏭")
        st.rerun()

with btn_r:
    if st.button("Connect  👉", use_container_width=True, key="connect_btn", type="primary"):
        # Add to connected list
        if not any(c["id"] == imp_id for c in st.session_state["connected"]):
            st.session_state["connected"].append({"id": imp_id, "score": score, "grade": grade})
        # Generate outreach
        st.session_state["show_outreach"] = True
        tone = st.session_state.get("tone", "professional")
        with st.spinner("Generating outreach email…"):
            out = _fetch_outreach(exp_id, match, tone)
        st.session_state["outreach"] = out
        st.toast(f"Connected with {imp_id}! Email ready.", icon="✅")
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  Outreach Email (shown after Connect)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("show_outreach") and st.session_state.get("outreach"):
    st.markdown("---")
    out = st.session_state["outreach"]

    hdr_col, tone_col = st.columns([1, 1])
    with hdr_col:
        st.markdown(
            f"""
            <div style='background:linear-gradient(135deg,#12282a,#0d1e20);
                        border:1px solid #2a4a4e;border-radius:14px;padding:14px 18px;margin-bottom:10px'>
              <h3 style='color:#c9a84c;margin:0 0 2px'>✉️ Outreach-Ready Email</h3>
              <p style='color:#4d5468;margin:0;font-size:.8em'>
                AI-generated for <b>{imp_id}</b>
                &nbsp;|&nbsp; via {out.get('generated_by','template')}
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tone_col:
        new_tone = st.radio(
            "Outreach Tone",
            ["professional", "friendly", "urgent"],
            index=["professional", "friendly", "urgent"].index(
                st.session_state.get("tone", "professional")
            ),
            horizontal=True,
            key="tone_radio",
        )
        if new_tone != st.session_state.get("tone"):
            st.session_state["tone"] = new_tone
            with st.spinner("Regenerating…"):
                out = _fetch_outreach(exp_id, match, new_tone)
            st.session_state["outreach"] = out
            st.rerun()

    if "error" in out:
        st.error(f"Email generation failed: {out['error']}")
    else:
        st.markdown(
            f"**Subject:** `{out.get('subject','')}`",
        )
        st.markdown(
            f"<div class='email-box'>{out.get('email_body','')}</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇ Download Email (.txt)",
            data=f"Subject: {out.get('subject','')}\n\n{out.get('email_body','')}",
            file_name=f"outreach_{imp_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )
