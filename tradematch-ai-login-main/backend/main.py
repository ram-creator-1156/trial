"""
TradeMatch AI — FastAPI Backend Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import match, exporter, importer
from backend.core.config import settings

app = FastAPI(
    title="TradeMatch AI",
    description="Swipe-to-Export Intelligent Matchmaking API",
    version="1.0.0",
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(match.router, prefix="/api/match", tags=["Matchmaking"])
app.include_router(exporter.router, prefix="/api/exporters", tags=["Exporters"])
app.include_router(importer.router, prefix="/api/importers", tags=["Importers"])


@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "healthy", "app": "TradeMatch AI"}
