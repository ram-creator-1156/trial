"""
Exporter API routes.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_exporters():
    """Return all exporters."""
    # TODO: load from data service
    return {"exporters": []}


@router.get("/{exporter_id}")
async def get_exporter(exporter_id: str):
    """Return a single exporter by ID."""
    # TODO: lookup exporter
    return {"exporter_id": exporter_id}
