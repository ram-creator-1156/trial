"""
Matchmaking API routes.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_matches():
    """Return top exporter-importer matches."""
    # TODO: integrate with matching service
    return {"matches": []}


@router.post("/swipe")
async def swipe_action(exporter_id: str, importer_id: str, direction: str):
    """Record a swipe (like / dislike) on a match card."""
    # TODO: persist swipe decision
    return {"status": "recorded", "exporter_id": exporter_id, "importer_id": importer_id, "direction": direction}
