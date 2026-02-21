"""
backend/routers/match.py
────────────────────────
FastAPI router for matchmaking endpoints.

Endpoints
─────────
GET  /api/matches          — Get top matches for a given importer (or global best)
POST /api/matches          — Same, via JSON body
POST /api/swipe            — Record a swipe action (like / dislike / superlike)
GET  /api/exporters        — List available exporters (with pagination)
GET  /api/importers        — List available importers (with pagination)
"""

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.schemas import (
    HealthResponse,
    MatchRequest,
    MatchResponse,
    MatchScore,
    SwipeAction,
)
from config.settings import get_settings, Settings
from models.matchmaker import MatchmakingEngine
from models.preprocessor import (
    build_tfidf_matrix,
    preprocess_exporters,
    preprocess_importers,
)
from utils.data_loader import get_exporters, get_importers, get_news_signals
from utils.logger import logger
from utils.news_signals import build_banned_countries, build_hs_sentiment_map

router = APIRouter(prefix="/api", tags=["matchmaking"])

# In-memory swipe log (replace with a real DB in production)
_swipe_log: list[dict] = []


# ── engine factory (singleton per process) ────────────────────────────────────

@lru_cache(maxsize=1)
def _get_engine() -> MatchmakingEngine:
    logger.info("Initialising MatchmakingEngine …")
    raw_exp = get_exporters()
    raw_imp = get_importers()
    raw_news = get_news_signals()

    exp_df, _ = preprocess_exporters(raw_exp)
    imp_df, _ = preprocess_importers(raw_imp)
    exp_tfidf, imp_tfidf, _ = build_tfidf_matrix(exp_df, imp_df)

    hs_sentiment   = build_hs_sentiment_map(raw_news)
    banned_countries = build_banned_countries(raw_news)

    return MatchmakingEngine(
        exporters_df=exp_df,
        importers_df=imp_df,
        exp_tfidf_matrix=exp_tfidf,
        imp_tfidf_matrix=imp_tfidf,
        hs_sentiment_map=hs_sentiment,
        banned_countries=banned_countries,
    )


def _engine_dep() -> MatchmakingEngine:
    return _get_engine()


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_importer_idx(engine: MatchmakingEngine, importer_id: str | None) -> int:
    if importer_id is None:
        return 0   # default to first importer
    id_col = next(
        (c for c in engine.imp_df.columns if "importer_id" in c or c == "id"),
        None,
    )
    if id_col is None:
        raise HTTPException(404, "importer_id column not found in dataset")
    matches = engine.imp_df[engine.imp_df[id_col].astype(str) == importer_id]
    if matches.empty:
        raise HTTPException(404, f"Importer '{importer_id}' not found")
    return int(matches.index[0])


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("/matches", response_model=MatchResponse)
async def get_matches(
    importer_id: str | None = Query(None, description="Filter by specific importer"),
    top_k: int = Query(10, ge=1, le=100),
    threshold: float = Query(0.0, ge=0.0, le=1.0),
    engine: MatchmakingEngine = Depends(_engine_dep),
):
    idx = _resolve_importer_idx(engine, importer_id)
    raw = engine.rank_for_importer(idx, top_k=top_k, threshold=threshold)
    matches = [MatchScore(**r.to_dict()) for r in raw]
    return MatchResponse(total_matches=len(matches), matches=matches)


@router.post("/matches", response_model=MatchResponse)
async def post_matches(
    body: MatchRequest,
    engine: MatchmakingEngine = Depends(_engine_dep),
):
    idx = _resolve_importer_idx(engine, body.importer_id)
    raw = engine.rank_for_importer(idx, top_k=body.top_k, threshold=body.threshold)
    matches = [MatchScore(**r.to_dict()) for r in raw]
    return MatchResponse(total_matches=len(matches), matches=matches)


@router.post("/swipe")
async def record_swipe(action: SwipeAction):
    entry = action.model_dump()
    _swipe_log.append(entry)
    logger.info(f"Swipe recorded: {entry}")
    return {"status": "ok", "recorded": entry}


@router.get("/swipes")
async def list_swipes(limit: int = Query(50, ge=1, le=500)):
    return {"total": len(_swipe_log), "swipes": _swipe_log[-limit:]}


@router.get("/exporters")
async def list_exporters(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    engine: MatchmakingEngine = Depends(_engine_dep),
):
    start = (page - 1) * page_size
    end   = start + page_size
    df    = engine.exp_df.iloc[start:end]
    return {
        "total":    len(engine.exp_df),
        "page":     page,
        "page_size": page_size,
        "data":     df.to_dict(orient="records"),
    }


@router.get("/importers")
async def list_importers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    engine: MatchmakingEngine = Depends(_engine_dep),
):
    start = (page - 1) * page_size
    end   = start + page_size
    df    = engine.imp_df.iloc[start:end]
    return {
        "total":    len(engine.imp_df),
        "page":     page,
        "page_size": page_size,
        "data":     df.to_dict(orient="records"),
    }
