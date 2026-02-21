"""
Importer API routes.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_importers():
    """Return all importers."""
    # TODO: load from data service
    return {"importers": []}


@router.get("/{importer_id}")
async def get_importer(importer_id: str):
    """Return a single importer by ID."""
    # TODO: lookup importer
    return {"importer_id": importer_id}
