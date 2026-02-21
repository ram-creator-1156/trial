"""
backend/main.py
═══════════════
FastAPI application for the TradeMatch LOC — Swipe-to-Export Intelligent
Matchmaking platform.

Endpoints
─────────
  GET  /health                         — Liveness / readiness probe
  GET  /api/matches/{exporter_id}      — Top-10 importer matches for a given
                                         exporter, with full Explainable
                                         Scoring JSON
  POST /api/simulate_real_time         — Inject a live-signal event and return
                                         reshuffled rankings (before / after)
  POST /api/generate_outreach          — Generate a personalised B2B outreach
                                         email via LangChain / GPT-4 (with a
                                         high-quality template fallback)

  POST /api/swipe                      — Record a like / dislike / superlike
  GET  /api/exporters                  — List all exporter IDs
  GET  /api/importers                  — List all importer IDs
  GET  /api/match_for_importer/{buyer_id} — Top-10 exporters for a buyer

  Run with:
      uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    ExporterProfile,
    HealthResponse,
    ImporterProfile,
    MatchResponse,
    MatchScore,
    OutreachRequest,
    OutreachResponse,
    RankDelta,
    SignalEvent,
    SimulateRequest,
    SimulateResponse,
    SwipeAction,
)

# ─────────────────────────────────────────────────────────────────────────────
#  App initialisation
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TradeMatch LOC — Matchmaking API",
    description=(
        "AI-powered Exporter ↔ Importer matchmaking with real-time signal "
        "adaptability and personalized outreach generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
#  Singleton engine loader — loads once, stays in memory
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_engine():
    """
    Load the Excel data, run the preprocessor, and initialise the
    MatchmakingEngine exactly once per process lifetime.
    Heavy — takes ~10 s on first call for 12K rows.
    """
    from config.settings import get_settings
    from models.matchmaker import MatchmakingEngine
    from utils.data_processor import load_and_process

    settings = get_settings()
    print("  [startup] Loading & processing data …")
    data = load_and_process(str(settings.excel_path), verbose=False)
    print(f"  [startup] Exporters={len(data.exporters):,}  Importers={len(data.importers):,}")
    engine = MatchmakingEngine(
        exporters_df=data.exporters,
        importers_df=data.importers,
        news_signals_df=data.news_signals,
        verbose=False,
    )
    print("  [startup] MatchmakingEngine ready ✓")
    return engine


# ─────────────────────────────────────────────────────────────────────────────
#  Helper — parse raw match dict → MatchScore schema
# ─────────────────────────────────────────────────────────────────────────────

def _to_match_score(row: dict) -> MatchScore:
    """Convert a raw dict from the engine into a validated MatchScore."""
    explanation = row.get("Explanation", "{}")
    if isinstance(explanation, str):
        try:
            explanation = json.loads(explanation)
        except json.JSONDecodeError:
            explanation = {}

    return MatchScore(
        exporter_id=str(row.get("Exporter_ID", "")),
        importer_buyer_id=str(row.get("Importer_Buyer_ID", "")),
        total_score=float(row.get("Total_Score", 0.0)),
        grade=str(row.get("Grade", "—")),
        p1_product=float(row.get("P1_Product", 0.0)),
        p2_geography=float(row.get("P2_Geography", 0.0)),
        p3_signals=float(row.get("P3_Signals", 0.0)),
        p4_activity=float(row.get("P4_Activity", 0.0)),
        explanation=explanation,
    )


def _clean_id(raw: str) -> str:
    """
    Normalise an ID that may have been serialised as a Python float.
    e.g.  '1.0'  -> '1'
          'EXP_001' -> 'EXP_001'  (unchanged)
    """
    raw = str(raw).strip()
    if raw.endswith(".0") and raw[:-2].isdigit():
        return raw[:-2]
    return raw


def _get_id_col(df, *candidates: str) -> str:
    """
    Return the first column name in df that matches one of the candidate
    names (case-insensitive).  Falls back to df.columns[0].
    Candidates tried in this order for exporters:
        exporter_id, exp_id, record_id
    Candidates tried for importers:
        buyer_id, importer_id, record_id
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return df.columns[0]



# ─────────────────────────────────────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health_check() -> HealthResponse:
    """
    Liveness & readiness probe.
    Returns dataset sizes and whether the engine has been loaded.
    """
    try:
        engine = _get_engine()
        return HealthResponse(
            status="ok",
            exporters_loaded=len(engine.exp),
            importers_loaded=len(engine.imp),
            engine_ready=True,
        )
    except Exception as exc:
        return HealthResponse(
            status=f"degraded: {exc}",
            exporters_loaded=0,
            importers_loaded=0,
            engine_ready=False,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 1 — GET /api/matches/{exporter_id}
# ═════════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/matches/{exporter_id}",
    response_model=MatchResponse,
    tags=["Matchmaking"],
    summary="Top importer matches for a given exporter (with Explainable Scoring)",
)
def get_matches_for_exporter(
    exporter_id: str,
    top_k: int = Query(10, ge=1, le=50, description="Number of matches to return"),
    threshold: float = Query(0.0, ge=0.0, le=100.0, description="Min Total_Score (0–100)"),
) -> MatchResponse:
    """
    Returns the **top-K importer matches** for a specific Exporter ID.

    Each match includes:
    - `total_score` (0–100) and letter `grade`
    - Per-pillar breakdown: `p1_product`, `p2_geography`, `p3_signals`, `p4_activity`
    - `explanation` — a full JSON dict with **human-readable reasons** for every
      sub-score, satisfying the *Explainable Scoring Logic* requirement.

    ### Score Pillars
    | Pillar | Weight | Criteria |
    |--------|--------|----------|
    | Product Compatibility | 30 pts | TF-IDF cosine, industry token overlap, certifications |
    | Geography Fit | 20 pts | Trade corridor score, Indian state bonus |
    | High-Intent Signals | 30 pts | Importer intent/hiring/funding, news bonus, risk penalty |
    | Trade Activity | 20 pts | Volume fit, revenue compat, buyer/exporter reliability |
    """
    engine = _get_engine()

    # Normalise the ID coming from the URL (may be '1.0' when dataset uses floats)
    clean_id = _clean_id(exporter_id)

    df: pd.DataFrame | None = None
    for try_id in dict.fromkeys([exporter_id, clean_id]):   # deduplicated, ordered
        try:
            df = engine.match_for_exporter(
                exporter_id=try_id,
                top_k=top_k,
                threshold=threshold / 100.0,
            )
            break
        except KeyError:
            continue
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    if df is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Exporter '{exporter_id}' not found in dataset. "
                f"Check /api/exporters for valid IDs."
            ),
        )

    if df.empty:
        return MatchResponse(exporter_id=exporter_id, total_matches=0, matches=[])

    matches = [_to_match_score(row) for _, row in df.iterrows()]
    return MatchResponse(
        exporter_id=exporter_id,
        total_matches=len(matches),
        matches=matches,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Bonus match endpoints kept for Streamlit UI compatibility
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/api/match_for_importer/{buyer_id}",
    response_model=MatchResponse,
    tags=["Matchmaking"],
    summary="Top exporter matches for a given importer",
)
def get_matches_for_importer(
    buyer_id: str,
    top_k: int = Query(10, ge=1, le=50),
    threshold: float = Query(0.0, ge=0.0, le=100.0),
) -> MatchResponse:
    """Returns the top-K exporter matches for a specific Importer Buyer_ID."""
    engine = _get_engine()
    try:
        df = engine.match_for_importer(
            buyer_id=buyer_id, top_k=top_k, threshold=threshold / 100.0
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Buyer '{buyer_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    matches = [_to_match_score(row) for _, row in df.iterrows()]
    return MatchResponse(exporter_id=None, total_matches=len(matches), matches=matches)


@app.get("/api/matches", response_model=MatchResponse, tags=["Matchmaking"],
         summary="Global top matches (random importer sample)")
def get_global_matches(
    top_k: int = Query(10, ge=1, le=50),
    threshold: float = Query(0.0, ge=0.0, le=100.0),
    n_sample: int = Query(200, ge=10, le=1000, description="Importer sample size"),
) -> MatchResponse:
    """Returns top-K global matches using a fast sampled strategy."""
    engine = _get_engine()
    try:
        df = engine.match_sample(
            n_importers=n_sample, top_k=top_k, threshold=threshold / 100.0
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    matches = [_to_match_score(row) for _, row in df.iterrows()]
    return MatchResponse(exporter_id=None, total_matches=len(matches), matches=matches)


# ═════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 2 — POST /api/simulate_real_time
# ═════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/simulate_real_time",
    response_model=SimulateResponse,
    tags=["Live Signals"],
    summary="Inject a live market signal and reshuffle match rankings in real time",
)
def simulate_real_time(request: SimulateRequest) -> SimulateResponse:
    """
    Triggers `simulate_live_signals()` to demonstrate **Real-Time Adaptability**.

    ### What happens
    1. Randomly selects `n_buyers` importers.
    2. Injects the chosen scenario's signal boost/penalty into their internal
       NumPy scoring arrays (e.g. Intent↑, Hiring↑, Risk↑, OrderTons×2).
    3. Re-runs the matchmaking engine and computes before-vs-after rankings.
    4. Restores all arrays to their original state (engine stays clean).
    5. Returns the full delta table plus a per-signal event log.

    ### Scenario keys
    | Key | Event |
    |----|-------|
    | `bulk_buy` | Bulk-purchase announcement |
    | `funding_round` | Series-B funding confirmed |
    | `tariff_cut` | Government tariff reduction |
    | `linkedin_spike` | Hiring / LinkedIn activity surge |
    | `geopolitical_risk` | Conflict / sanction risk escalation |
    | `new_trade_deal` | Bilateral trade agreement signed |
    | `supply_shock` | Supply chain disruption — urgent sourcing |
    """
    from utils.live_signals import SCENARIOS, simulate_live_signals

    # Validate scenario key
    if request.scenario not in SCENARIOS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown scenario '{request.scenario}'. "
                f"Valid keys: {list(SCENARIOS.keys())}"
            ),
        )

    engine = _get_engine()

    try:
        result = simulate_live_signals(
            engine=engine,
            n_buyers=request.n_buyers,
            seed=request.seed if request.seed > 0 else None,
            scenario=request.scenario,
            top_k=request.top_k,
            n_sample_importers=300,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # ── Convert delta_df rows → RankDelta objects ─────────────────────────────
    col_map = {
        "Exporter_ID":       "exporter_id",
        "Importer_Buyer_ID": "importer_buyer_id",
        "Score_Before":      "score_before",
        "Score_After":       "score_after",
        "Score_Delta":       "score_delta",
        "Rank_Change":       "rank_change",
        "Movement":          "movement",
        "Signal_Injected":   "signal_injected",
    }
    delta_rows: list[RankDelta] = []
    for _, row in result.delta_df.iterrows():
        delta_rows.append(RankDelta(
            exporter_id=str(row.get("Exporter_ID", "")),
            importer_buyer_id=str(row.get("Importer_Buyer_ID", "")),
            score_before=float(row.get("Score_Before", 0.0)),
            score_after=float(row.get("Score_After", 0.0)),
            score_delta=float(row.get("Score_Delta", 0.0)),
            rank_change=int(row.get("Rank_Change", 0)),
            movement=_clean_movement(str(row.get("Movement", "Flat"))),
            signal_injected=bool(row.get("Signal_Injected", False)),
        ))

    top_movers = sorted(delta_rows, key=lambda r: r.rank_change, reverse=True)[:5]
    top_losers  = sorted(delta_rows, key=lambda r: r.rank_change)[:5]

    # ── Convert event_log → SignalEvent objects ───────────────────────────────
    events = [
        SignalEvent(
            buyer_id=e["buyer_id"],
            signal=e["signal"],
            before=float(e["before"]),
            after=float(e["after"]),
            delta=float(e["delta"]),
        )
        for e in result.event_log
    ]

    return SimulateResponse(
        scenario=result.scenario,
        scenario_icon=result.scenario_icon,
        scenario_desc=result.scenario_desc,
        affected_buyers=result.affected_buyers,
        n_affected=result.n_affected,
        timestamp=result.timestamp,
        total_pairs=len(delta_rows),
        pairs_moved_up=sum(1 for r in delta_rows if r.rank_change > 0),
        pairs_moved_down=sum(1 for r in delta_rows if r.rank_change < 0),
        top_movers=top_movers,
        top_losers=top_losers,
        event_log=events,
    )


def _clean_movement(raw: str) -> str:
    """Normalise movement string to 'Up' | 'Down' | 'Flat' (strip emoji)."""
    raw = raw.strip()
    if "Up" in raw or "up" in raw:
        return "Up"
    if "Down" in raw or "down" in raw:
        return "Down"
    return "Flat"


# ═════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 3 — POST /api/generate_outreach
# ═════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/generate_outreach",
    response_model=OutreachResponse,
    tags=["Outreach AI"],
    summary="Generate a personalised B2B outreach email (LangChain / GPT-4 or template)",
)
def generate_outreach(request: OutreachRequest) -> OutreachResponse:
    """
    Produces a **highly personalised, outreach-ready B2B email** tailored to
    the specific exporter ↔ importer match.

    ### Generation modes
    | Mode | When | What |
    |------|------|------|
    | **LLM (GPT-4)** | `use_llm=true` + `OPENAI_API_KEY` set | LangChain `ChatOpenAI` prompt chain |
    | **Template** | `use_llm=false` or no API key | Deterministic high-quality template |

    ### Personalisation layers
    - Exporter: company, industry, state, certifications, capacity, match grade
    - Importer: company, country, industry, avg order size, response probability
    - Match: total score, pillar strengths pulled from Explainable Scoring JSON
    - Tone: `professional` | `friendly` | `urgent`

    The template fallback is indistinguishable in quality for demo purposes and
    requires zero external API calls.
    """
    # Determine strongest match pillar for personalisation
    pillar_names = {
        "1_product_compatibility": "product alignment",
        "2_geography_fit":         "geographic proximity",
        "3_high_intent_signals":   "high purchase intent",
        "4_trade_activity":        "trade activity compatibility",
    }
    strongest_pillar = "strong compatibility"
    if request.explanation:
        best_key = max(
            [k for k in pillar_names if k in request.explanation],
            key=lambda k: request.explanation[k].get("score", 0),
            default=None,
        )
        if best_key:
            strongest_pillar = pillar_names[best_key]

    # ── Try LLM path first ────────────────────────────────────────────────────
    if request.use_llm:
        try:
            subject, body = _generate_with_llm(request, strongest_pillar)
            generated_by = "llm"
        except Exception as llm_exc:
            print(f"[outreach] LLM call failed ({llm_exc}), falling back to template.")
            subject, body = _generate_with_template(request, strongest_pillar)
            generated_by = "template (llm_failed)"
    else:
        subject, body = _generate_with_template(request, strongest_pillar)
        generated_by = "template"

    return OutreachResponse(
        exporter_id=request.exporter.exporter_id,
        buyer_id=request.importer.buyer_id,
        subject=subject,
        email_body=body,
        generated_by=generated_by,
        tone=request.tone,
        match_score=request.match_score,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Outreach — LangChain / GPT-4 path
# ─────────────────────────────────────────────────────────────────────────────

def _generate_with_llm(request: OutreachRequest, strongest_pillar: str) -> tuple[str, str]:
    """
    Uses LangChain's ChatOpenAI to generate the email.
    Raises any exception upward so the caller can fall back to template.
    """
    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # Build a rich context string from the match data
    context = _build_context_str(request, strongest_pillar)

    # LangChain prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are TradeMatch AI, an expert in international B2B trade outreach. "
                "Your emails are concise, credible, and conversion-focused. "
                "Always write in {tone} tone. Never use filler phrases like 'I hope this email finds you well'."
            ),
        ),
        (
            "human",
            (
                "Write a highly personalised B2B outreach email based on this match context:\n\n"
                "{context}\n\n"
                "Requirements:\n"
                "- Subject line: compelling, specific, under 60 characters\n"
                "- Body: 150-200 words, 3 short paragraphs\n"
                "- Para 1: Why this match is uniquely relevant (cite the match score and top pillar)\n"
                "- Para 2: What the exporter offers specifically for the importer's needs\n"
                "- Para 3: Clear, low-friction CTA (suggest a specific meeting time or a brief call)\n"
                "- Sign off as: [Exporter Company] via TradeMatch LOC Platform\n\n"
                "Return ONLY: SUBJECT: <line>\\n\\n<email body>"
            ),
        ),
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",           # cost-efficient for hackathon
        temperature=0.7,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chain = prompt | llm
    response = chain.invoke({"tone": request.tone, "context": context})
    raw_text: str = response.content.strip()

    # Parse subject / body
    if "SUBJECT:" in raw_text:
        parts = raw_text.split("\n\n", 1)
        subject = parts[0].replace("SUBJECT:", "").strip()
        body    = parts[1].strip() if len(parts) > 1 else raw_text
    else:
        lines   = raw_text.split("\n", 1)
        subject = lines[0].strip()
        body    = lines[1].strip() if len(lines) > 1 else raw_text

    return subject, body


# ─────────────────────────────────────────────────────────────────────────────
#  Outreach — Template fallback (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_str(request: OutreachRequest, strongest_pillar: str) -> str:
    """Serialise the outreach request into a human-readable context block."""
    exp = request.exporter
    imp = request.importer
    score = request.match_score
    lines = [
        f"Exporter: {exp.company_name} (ID: {exp.exporter_id})",
        f"  Industry: {exp.industry} | State: {exp.state}",
        f"  Certifications: {exp.certifications or 'N/A'}",
        f"  Capacity: {f'{exp.capacity_tons:,.0f} tons' if exp.capacity_tons else 'N/A'}",
        f"  Revenue: {f'${exp.revenue_usd:,.0f}' if exp.revenue_usd else 'N/A'}",
        "",
        f"Importer: {imp.company_name} (ID: {imp.buyer_id})",
        f"  Industry: {imp.industry} | Country: {imp.country}",
        f"  Avg Order: {f'{imp.avg_order_tons:,.0f} tons' if imp.avg_order_tons else 'N/A'}",
        f"  Response Probability: {f'{imp.response_probability:.0%}' if imp.response_probability else 'N/A'}",
        "",
        f"Match Score: {f'{score:.1f}/100' if score is not None else 'N/A'}",
        f"Strongest Compatibility Signal: {strongest_pillar}",
    ]
    return "\n".join(lines)


def _generate_with_template(
    request: OutreachRequest, strongest_pillar: str
) -> tuple[str, str]:
    """
    Deterministic, high-quality template-based email generator.
    Personalises across exporter, importer, match score, and tone.
    """
    exp   = request.exporter
    imp   = request.importer
    score = request.match_score
    tone  = request.tone

    # ── Opener strategy by tone ───────────────────────────────────────────────
    openers = {
        "professional": (
            f"I am reaching out on behalf of {exp.company_name}, a leading "
            f"{exp.industry} supplier based in {exp.state}, India."
        ),
        "friendly": (
            f"Hi there — I came across {imp.company_name} on the TradeMatch LOC "
            f"platform and knew we had to connect."
        ),
        "urgent": (
            f"Given current market dynamics in {imp.industry}, I wanted to reach "
            f"out immediately regarding a time-sensitive partnership opportunity."
        ),
    }

    # ── Pillar-specific value proposition ────────────────────────────────────
    pillar_props = {
        "product alignment": (
            f"Our {exp.industry} product lines map precisely onto your sourcing requirements, "
            f"with shared certification standards ensuring zero integration friction."
        ),
        "geographic proximity": (
            f"Our India-based operations give you access to one of the fastest growing "
            f"export hubs, with direct logistics routes to {imp.country} already established."
        ),
        "high purchase intent": (
            f"Our platform's real-time signals indicate your team is actively seeking "
            f"suppliers — we are in a strong position to fulfil your current demand cycle."
        ),
        "trade activity compatibility": (
            f"Our export volumes and capacity align closely with your average order profile"
            + (f" of {imp.avg_order_tons:,.0f} tons" if imp.avg_order_tons else "")
            + ", making us a scalable, low-risk supply partner."
        ),
        "strong compatibility": (
            f"Our comprehensive compatibility assessment across product, geography, "
            f"and trade activity signals places this partnership in our top-tier category."
        ),
    }

    # ── CTA by tone ───────────────────────────────────────────────────────────
    ctas = {
        "professional": (
            f"I would welcome the opportunity for a 20-minute introductory call at your "
            f"convenience. Would Thursday or Friday this week work for your team?"
        ),
        "friendly": (
            f"Would love to jump on a quick call — are you free for 15 minutes this week? "
            f"I'm confident we can make something great happen."
        ),
        "urgent": (
            f"Given the current window of opportunity, could we schedule a call in the "
            f"next 48 hours? I can make myself available at your earliest convenience."
        ),
    }

    # ── Certification line ────────────────────────────────────────────────────
    cert_line = ""
    if exp.certifications and exp.certifications.lower() not in ("nan", "unknown", "none"):
        cert_line = f"We hold {exp.certifications} certification, "

    # ── Score line ────────────────────────────────────────────────────────────
    score_line = ""
    if score is not None:
        score_line = (
            f"Our AI-powered matchmaking platform scored this pairing "
            f"**{score:.0f}/100** — driven primarily by our {strongest_pillar}. "
        )

    # ── Capacity line ─────────────────────────────────────────────────────────
    capacity_line = ""
    if exp.capacity_tons:
        capacity_line = (
            f"{cert_line}with a manufacturing capacity of {exp.capacity_tons:,.0f} tons "
            f"annually, well-positioned to serve your import needs at scale."
        )
    else:
        capacity_line = (
            f"{cert_line}a well-established exporter with a proven track record "
            f"in {exp.industry}."
        )

    # ── Assemble ──────────────────────────────────────────────────────────────
    subject = (
        f"Partnership Opportunity: {exp.company_name} × {imp.company_name} "
        f"— {strongest_pillar.title()} Match"
    )
    # Keep subject under 80 chars
    if len(subject) > 80:
        subject = f"{exp.company_name} — {strongest_pillar.title()} Opportunity for {imp.country}"

    body = textwrap.dedent(f"""\
        Dear {imp.company_name} Procurement Team,

        {openers[tone]} {score_line}

        {pillar_props.get(strongest_pillar, pillar_props['strong compatibility'])} We are {capacity_line}

        {ctas[tone]}

        Looking forward to exploring this opportunity together.

        Warm regards,
        {exp.company_name}
        {exp.state}, India
        via TradeMatch LOC Platform · Swipe-to-Export AI Matchmaking
    """).strip()

    return subject, body


# ─────────────────────────────────────────────────────────────────────────────
#  Utility endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/swipe", tags=["Utility"], summary="Record a swipe action")
def record_swipe(action: SwipeAction) -> dict:
    """
    Persists a like / dislike / superlike action.
    (In production this writes to a database; here we log and return confirmation.)
    """
    print(
        f"[swipe] exporter={action.exporter_id}  "
        f"importer={action.importer_id}  action={action.action}"
    )
    return {
        "status": "recorded",
        "exporter_id": action.exporter_id,
        "importer_id": action.importer_id,
        "action": action.action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/exporters", tags=["Utility"], summary="List all exporter IDs")
def list_exporters(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    """Returns a paginated list of exporter IDs from the loaded dataset."""
    engine = _get_engine()
    id_col = _get_id_col(engine.exp, "exporter_id", "exp_id", "record_id")
    # Clean float IDs (1.0 -> 1) so the dropdown sends valid strings
    ids = [_clean_id(v) for v in engine.exp[id_col].astype(str).tolist()]
    return {
        "total": len(ids),
        "offset": offset,
        "limit": limit,
        "exporter_ids": ids[offset: offset + limit],
    }


@app.get("/api/importers", tags=["Utility"], summary="List all importer IDs")
def list_importers(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    """Returns a paginated list of importer Buyer IDs from the loaded dataset."""
    engine = _get_engine()
    id_col = _get_id_col(engine.imp, "buyer_id", "importer_id", "record_id")
    # Clean float IDs (1.0 -> 1)
    ids = [_clean_id(v) for v in engine.imp[id_col].astype(str).tolist()]
    return {
        "total": len(ids),
        "offset": offset,
        "limit": limit,
        "buyer_ids": ids[offset: offset + limit],
    }
