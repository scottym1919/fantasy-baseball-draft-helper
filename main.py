"""
Fantasy Baseball Draft Helper — Main Application Entry Point

A statistically-driven draft optimization tool for rotisserie leagues.

Run with:
    python main.py
    # or
    uvicorn main:app --reload --port 8000
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from backend.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# Create FastAPI app
app = FastAPI(
    title="Fantasy Baseball Draft Helper",
    description="Rotisserie draft optimizer with SGP analysis, Monte Carlo simulation, and ESPN integration",
    version="0.1.0",
)

# Mount static files and templates
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "frontend" / "templates"))

# Include API routes
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main draft helper UI."""
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
