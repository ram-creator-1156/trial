"""
backend/schemas.py
──────────────────
Pydantic v2 request / response models for all FastAPI endpoints.

Sections
────────
  1. Core match models         — MatchRequest, MatchScore, MatchResponse
  2. Simulation models         — SimulateRequest, SimulateResponse
  3. Outreach models           — OutreachRequest, OutreachResponse
  4. Shared / util models      — HealthResponse, SwipeAction
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
#  1. Core match models
# ─────────────────────────────────────────────────────────────────────────────

class MatchRequest(BaseModel):
    """Body schema for POST /api/matches."""
    importer_id: Optional[str] = Field(
        None,
        description="Specific importer Buyer_ID to match against. "
                    "If omitted, returns global top matches.",
    )
    top_k: int = Field(10, ge=1, le=100, description="Max number of matches to return")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum match score (0–1)")


class PillarDetail(BaseModel):
    """One pillar inside the Explainable Scoring JSON."""
    score: float = Field(..., description="Normalised pillar score (0–1)")
    max_pts: int  = Field(..., description="Maximum points this pillar contributes")
    reasons: list[str] = Field(default_factory=list, description="Human-readable reasons")


class ExplainableScoring(BaseModel):
    """Full explainability breakdown attached to every match."""
    product_compatibility: PillarDetail = Field(alias="1_product_compatibility")
    geography_fit:         PillarDetail = Field(alias="2_geography_fit")
    high_intent_signals:   PillarDetail = Field(alias="3_high_intent_signals")
    trade_activity:        PillarDetail = Field(alias="4_trade_activity")

    model_config = {"populate_by_name": True}


class MatchScore(BaseModel):
    """Single exporter–importer match result, including all pillar scores."""
    exporter_id:       str
    importer_buyer_id: str
    total_score:       float = Field(..., description="Weighted total score out of 100")
    grade:             str   = Field(..., description="Letter grade: A+, A, B+, B, C, D")
    p1_product:        float = Field(..., description="Product compatibility score (max 30)")
    p2_geography:      float = Field(..., description="Geography fit score (max 20)")
    p3_signals:        float = Field(..., description="High-intent signals score (max 30)")
    p4_activity:       float = Field(..., description="Trade activity score (max 20)")
    explanation:       dict  = Field(..., description="Full explainable scoring JSON per pillar")


class MatchResponse(BaseModel):
    """Response envelope for all match endpoints."""
    exporter_id:   Optional[str] = None
    total_matches: int
    matches:       list[MatchScore]


# ─────────────────────────────────────────────────────────────────────────────
#  2. Real-time simulation models
# ─────────────────────────────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    """Body for POST /api/simulate_real_time."""
    scenario: str = Field(
        "bulk_buy",
        description=(
            "Signal scenario key. One of: bulk_buy, funding_round, tariff_cut, "
            "linkedin_spike, geopolitical_risk, new_trade_deal, supply_shock"
        ),
    )
    n_buyers: int = Field(5, ge=1, le=20, description="Number of importers to inject signal into")
    top_k:    int = Field(15, ge=1, le=50,  description="Top-K results to return in before/after")
    seed:     int = Field(42, ge=0, description="Random seed (0 = fully random)")


class SignalEvent(BaseModel):
    """A single per-buyer signal mutation recorded during simulation."""
    buyer_id: str
    signal:   str
    before:   float
    after:    float
    delta:    float


class RankDelta(BaseModel):
    """One row of the before-vs-after ranking comparison."""
    exporter_id:       str
    importer_buyer_id: str
    score_before:      float
    score_after:       float
    score_delta:       float
    rank_change:       int
    movement:          str   # "Up", "Down", "Flat"
    signal_injected:   bool


class SimulateResponse(BaseModel):
    """Response for POST /api/simulate_real_time."""
    scenario:          str
    scenario_icon:     str
    scenario_desc:     str
    affected_buyers:   list[str]
    n_affected:        int
    timestamp:         str
    total_pairs:       int
    pairs_moved_up:    int
    pairs_moved_down:  int
    top_movers:        list[RankDelta]
    top_losers:        list[RankDelta]
    event_log:         list[SignalEvent]


# ─────────────────────────────────────────────────────────────────────────────
#  3. Outreach email generation models
# ─────────────────────────────────────────────────────────────────────────────

class ExporterProfile(BaseModel):
    """Minimal exporter profile required to generate outreach email."""
    exporter_id:   str
    company_name:  Optional[str] = "the exporting company"
    industry:      Optional[str] = "General Trade"
    state:         Optional[str] = "India"
    certifications: Optional[str] = None
    capacity_tons: Optional[float] = None
    revenue_usd:   Optional[float] = None
    intent_score:  Optional[float] = None


class ImporterProfile(BaseModel):
    """Minimal importer profile required to generate outreach email."""
    buyer_id:       str
    company_name:   Optional[str] = "the importing company"
    country:        Optional[str] = "International"
    industry:       Optional[str] = "General Trade"
    avg_order_tons: Optional[float] = None
    revenue_usd:    Optional[float] = None
    certifications: Optional[str] = None
    response_probability: Optional[float] = None


class OutreachRequest(BaseModel):
    """Body for POST /api/generate_outreach."""
    exporter:    ExporterProfile
    importer:    ImporterProfile
    match_score: Optional[float] = Field(None, description="Total match score (0–100)")
    explanation: Optional[dict]  = Field(None, description="Explainable scoring dict from /api/matches")
    tone:        str = Field(
        "professional",
        description="Email tone: professional | friendly | urgent",
        pattern="^(professional|friendly|urgent)$",
    )
    use_llm:     bool = Field(
        False,
        description="If True and OPENAI_API_KEY is set, use GPT-4 via LangChain. "
                    "Otherwise returns a high-quality template-generated email.",
    )


class OutreachResponse(BaseModel):
    """Response for POST /api/generate_outreach."""
    exporter_id:  str
    buyer_id:     str
    subject:      str
    email_body:   str
    generated_by: str   # "llm" | "template"
    tone:         str
    match_score:  Optional[float] = None
    timestamp:    str


# ─────────────────────────────────────────────────────────────────────────────
#  4. Shared / utility models
# ─────────────────────────────────────────────────────────────────────────────

class SwipeAction(BaseModel):
    """Body for POST /api/swipe."""
    exporter_id: str
    importer_id: str
    action: str = Field(..., pattern="^(like|dislike|superlike)$")


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status:            str
    exporters_loaded:  int
    importers_loaded:  int
    engine_ready:      bool
    version:           str = "1.0.0"
