"""
FastAPI routes for the draft helper API.

Provides REST endpoints for:
  - Uploading projections (CSV files)
  - Configuring league settings
  - Getting draft recommendations
  - Recording draft picks
  - ESPN integration (connect, poll draft)
  - Pitcher streaming analysis

Plus a WebSocket endpoint for real-time draft updates.
"""

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Optional

from fastapi import (
    APIRouter, FastAPI, WebSocket, WebSocketDisconnect,
    UploadFile, File, Form, HTTPException,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..engine.config import LeagueSettings
from ..engine.draft_optimizer import DraftOptimizer
from ..data.projections import (
    load_hitter_projections, load_pitcher_projections,
    combine_projections, add_adp_data, ProjectionSystem,
)
from ..espn.client import ESPNClient, ESPNCredentials, match_espn_to_projections

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Global state ──────────────────────────────────────────────────────
# In a production app these would be in a proper state store.
# For a single-user draft tool, module-level state is fine.

_optimizer: Optional[DraftOptimizer] = None
_espn_client: Optional[ESPNClient] = None
_espn_player_map: dict = {}  # espn_id -> projection_player_id
_ws_clients: list[WebSocket] = []
_draft_poll_task: Optional[asyncio.Task] = None


# ── Pydantic models ─────────────────────────────────────────────────

class LeagueSettingsRequest(BaseModel):
    num_teams: int = 12
    my_team_idx: int = 0
    num_rounds: int = 23
    snake_draft: bool = True
    roster_slots: Optional[dict[str, int]] = None
    hitting_categories: Optional[list[str]] = None
    pitching_categories: Optional[list[str]] = None


class ESPNConnectRequest(BaseModel):
    league_id: int
    season: int = 2025
    espn_s2: str
    swid: str


class RecordPickRequest(BaseModel):
    team_idx: int
    player_id: str
    player_name: Optional[str] = ""


class ManualPickRequest(BaseModel):
    """For manually entering a pick by player name (fuzzy search)."""
    team_idx: int
    player_name: str


# ── Setup endpoint ────────────────────────────────────────────────────

@router.post("/api/setup")
async def setup_league(req: LeagueSettingsRequest):
    """Initialize or update league settings."""
    global _optimizer

    kwargs = {}
    if req.roster_slots:
        kwargs["roster_slots"] = req.roster_slots
    if req.hitting_categories:
        kwargs["hitting_categories"] = req.hitting_categories
    if req.pitching_categories:
        kwargs["pitching_categories"] = req.pitching_categories

    settings = LeagueSettings(
        num_teams=req.num_teams,
        num_rounds=req.num_rounds,
        snake_draft=req.snake_draft,
        **kwargs,
    )

    _optimizer = DraftOptimizer(
        settings=settings,
        my_team_idx=req.my_team_idx,
    )

    return {"status": "ok", "settings": {
        "num_teams": settings.num_teams,
        "num_rounds": settings.num_rounds,
        "roster_size": settings.total_roster_size,
        "categories": settings.all_categories,
    }}


# ── Projection upload ────────────────────────────────────────────────

@router.post("/api/projections/upload")
async def upload_projections(
    hitters_file: Optional[UploadFile] = File(None),
    pitchers_file: Optional[UploadFile] = File(None),
    adp_file: Optional[UploadFile] = File(None),
    projection_system: str = Form("custom"),
):
    """
    Upload projection CSV files.

    Accepts separate hitter and pitcher files (FanGraphs format).
    Optionally upload an ADP file for draft position data.
    """
    global _optimizer
    if _optimizer is None:
        raise HTTPException(400, "Call /api/setup first to configure the league")

    system = ProjectionSystem(projection_system)
    hitters = None
    pitchers = None

    if hitters_file:
        content = await hitters_file.read()
        hitters = load_hitter_projections(content, system)
        logger.info("Loaded %d hitter projections", len(hitters))

    if pitchers_file:
        content = await pitchers_file.read()
        pitchers = load_pitcher_projections(content, system)
        logger.info("Loaded %d pitcher projections", len(pitchers))

    if hitters is None and pitchers is None:
        raise HTTPException(400, "Provide at least one projection file")

    if hitters is None:
        import pandas as pd
        hitters = pd.DataFrame()
    if pitchers is None:
        import pandas as pd
        pitchers = pd.DataFrame()

    combined = combine_projections(hitters, pitchers)

    # Add ADP data if provided
    if adp_file:
        adp_content = await adp_file.read()
        import pandas as pd
        import io
        adp_df = pd.read_csv(io.BytesIO(adp_content))
        combined = add_adp_data(combined, adp_df)

    _optimizer.load_projections(combined, hitters, pitchers)

    return {
        "status": "ok",
        "players_loaded": len(combined),
        "hitters": len(hitters),
        "pitchers": len(pitchers),
        "has_adp": "adp" in combined.columns,
    }


# ── Recommendations ──────────────────────────────────────────────────

@router.get("/api/recommendations")
async def get_recommendations(
    top_n: int = 25,
    monte_carlo: bool = False,
    simulations: int = 500,
):
    """
    Get ranked draft recommendations for the current pick.

    Set monte_carlo=true for more accurate (but slower) simulation-based
    recommendations. Default uses fast analytical ranking.
    """
    if _optimizer is None or _optimizer.player_pool is None:
        raise HTTPException(400, "No projections loaded. Upload data first.")

    recs = _optimizer.get_recommendations(
        top_n=top_n,
        use_monte_carlo=monte_carlo,
        n_simulations=simulations,
    )

    return {
        "pick": _optimizer.current_pick + 1,
        "round": _optimizer.current_round,
        "is_my_pick": _optimizer.is_my_pick,
        "recommendations": [asdict(r) for r in recs],
    }


# ── Draft picks ──────────────────────────────────────────────────────

@router.post("/api/pick")
async def record_pick(req: RecordPickRequest):
    """Record a draft pick (from any team)."""
    if _optimizer is None:
        raise HTTPException(400, "No draft in progress. Call /api/setup first.")

    _optimizer.record_pick(req.team_idx, req.player_id, req.player_name or "")

    # Notify WebSocket clients
    await _broadcast({
        "type": "pick",
        "pick": _optimizer.current_pick,
        "round": _optimizer.current_round,
        "team_idx": req.team_idx,
        "player_id": req.player_id,
        "player_name": req.player_name,
    })

    return {"status": "ok", "current_pick": _optimizer.current_pick + 1}


@router.post("/api/pick/search")
async def record_pick_by_name(req: ManualPickRequest):
    """Record a pick by searching for a player name (fuzzy match)."""
    if _optimizer is None or _optimizer.player_pool is None:
        raise HTTPException(400, "No draft configured")

    pool = _optimizer.player_pool
    available = pool[pool.index.isin(_optimizer.available_player_ids)]

    # Search by name (case-insensitive contains)
    search = req.player_name.lower()
    matches = available[available["name"].str.lower().str.contains(search, na=False)]

    if len(matches) == 0:
        raise HTTPException(404, f"No available player matching '{req.player_name}'")

    # Take the best match (shortest name that contains the search string)
    best = matches.iloc[0]
    player_id = str(best.name)
    player_name = best.get("name", player_id)

    _optimizer.record_pick(req.team_idx, player_id, player_name)

    await _broadcast({
        "type": "pick",
        "pick": _optimizer.current_pick,
        "round": _optimizer.current_round,
        "team_idx": req.team_idx,
        "player_id": player_id,
        "player_name": player_name,
    })

    return {
        "status": "ok",
        "player_id": player_id,
        "player_name": player_name,
        "current_pick": _optimizer.current_pick + 1,
    }


@router.post("/api/pick/undo")
async def undo_pick():
    """Undo the most recent draft pick."""
    if _optimizer is None:
        raise HTTPException(400, "No draft in progress")

    _optimizer.undo_last_pick()
    await _broadcast({"type": "undo", "current_pick": _optimizer.current_pick + 1})

    return {"status": "ok", "current_pick": _optimizer.current_pick + 1}


# ── Draft summary ────────────────────────────────────────────────────

@router.get("/api/draft/summary")
async def get_draft_summary():
    """Get current draft state summary."""
    if _optimizer is None:
        raise HTTPException(400, "No draft configured")
    return _optimizer.get_draft_summary()


@router.get("/api/draft/my-roster")
async def get_my_roster():
    """Get my team's current roster with projections."""
    if _optimizer is None or _optimizer.player_pool is None:
        raise HTTPException(400, "No draft configured")

    pool = _optimizer.player_pool
    my_ids = _optimizer.my_roster
    roster = pool[pool.index.isin(my_ids)]

    players = []
    for _, row in roster.iterrows():
        players.append({
            "player_id": str(row.get("player_id", row.name)),
            "name": row.get("name", ""),
            "position": row.get("position", ""),
            "team": row.get("team", ""),
            "sgp_total": round(float(row.get("SGP_total", 0)), 2),
            "var": round(float(row.get("VAR", 0)), 2),
        })

    return {"roster": players, "total_sgp": round(roster["SGP_total"].sum(), 2)}


@router.get("/api/draft/picks")
async def get_all_picks():
    """Get all draft picks made so far."""
    if _optimizer is None:
        return {"picks": []}
    return {"picks": [asdict(p) for p in _optimizer.picks]}


# ── Pitcher streaming ────────────────────────────────────────────────

@router.get("/api/streaming")
async def get_streaming_analysis():
    """Analyze pitcher streaming value for my roster."""
    if _optimizer is None or _optimizer.player_pool is None:
        raise HTTPException(400, "No draft configured")
    return _optimizer.get_streaming_analysis()


# ── Player search ────────────────────────────────────────────────────

@router.get("/api/players/search")
async def search_players(q: str, limit: int = 20):
    """Search available players by name."""
    if _optimizer is None or _optimizer.player_pool is None:
        raise HTTPException(400, "No projections loaded")

    pool = _optimizer.player_pool
    available = pool[pool.index.isin(_optimizer.available_player_ids)]
    matches = available[available["name"].str.lower().str.contains(q.lower(), na=False)]

    results = []
    for _, row in matches.head(limit).iterrows():
        results.append({
            "player_id": str(row.get("player_id", row.name)),
            "name": row.get("name", ""),
            "position": row.get("position", ""),
            "team": row.get("team", ""),
            "sgp_total": round(float(row.get("SGP_total", 0)), 2),
            "var": round(float(row.get("VAR", 0)), 2),
            "adp": round(float(row.get("adp", 999)), 1),
        })

    return {"results": results}


# ── ESPN integration ─────────────────────────────────────────────────

@router.post("/api/espn/connect")
async def connect_espn(req: ESPNConnectRequest):
    """Connect to an ESPN league using cookies."""
    global _espn_client, _espn_player_map

    creds = ESPNCredentials(espn_s2=req.espn_s2, swid=req.swid)
    _espn_client = ESPNClient(
        league_id=req.league_id,
        season=req.season,
        credentials=creds,
    )

    try:
        league_info = await _espn_client.get_league_info()
    except Exception as e:
        raise HTTPException(400, f"Failed to connect to ESPN: {e}")

    # Try to fetch player data for ID mapping
    try:
        espn_players = await _espn_client.get_players()
        if _optimizer and _optimizer.player_pool is not None:
            _espn_player_map = match_espn_to_projections(
                espn_players, _optimizer.player_pool
            )
            logger.info("Matched %d ESPN players to projections", len(_espn_player_map))
    except Exception as e:
        logger.warning("Could not fetch ESPN players: %s", e)

    return {
        "status": "ok",
        "league_name": league_info.name,
        "num_teams": league_info.num_teams,
        "teams": league_info.team_names,
        "scoring_type": league_info.scoring_type,
    }


@router.post("/api/espn/start-polling")
async def start_espn_polling(interval: float = 8.0):
    """Start polling ESPN for live draft picks."""
    global _draft_poll_task

    if _espn_client is None:
        raise HTTPException(400, "Connect to ESPN first via /api/espn/connect")

    if _draft_poll_task and not _draft_poll_task.done():
        return {"status": "already_polling"}

    async def on_new_picks(new_picks, total):
        for pick in new_picks:
            # Map ESPN player ID to our projection player ID
            proj_id = _espn_player_map.get(pick.player_id, str(pick.player_id))

            if _optimizer:
                _optimizer.record_pick(
                    team_idx=pick.team_id,
                    player_id=proj_id,
                    player_name=pick.player_name,
                )

            await _broadcast({
                "type": "espn_pick",
                "overall_pick": pick.overall_pick,
                "team_id": pick.team_id,
                "player_name": pick.player_name,
                "player_id": proj_id,
            })

    _draft_poll_task = asyncio.create_task(
        _espn_client.poll_draft(on_new_picks, poll_interval=interval)
    )

    return {"status": "polling_started", "interval": interval}


@router.post("/api/espn/stop-polling")
async def stop_espn_polling():
    """Stop polling ESPN for draft picks."""
    global _draft_poll_task
    if _draft_poll_task and not _draft_poll_task.done():
        _draft_poll_task.cancel()
    return {"status": "polling_stopped"}


@router.get("/api/espn/draft-status")
async def espn_draft_status():
    """Check ESPN draft status."""
    if _espn_client is None:
        raise HTTPException(400, "Not connected to ESPN")
    try:
        status = await _espn_client.get_draft_status()
        return status
    except Exception as e:
        raise HTTPException(500, f"ESPN error: {e}")


# ── WebSocket ────────────────────────────────────────────────────────

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time draft updates.

    Clients receive JSON messages with types:
      - "pick": New draft pick recorded
      - "espn_pick": Pick detected from ESPN polling
      - "undo": Pick was undone
      - "recommendation_update": New recommendations available
    """
    await websocket.accept()
    _ws_clients.append(websocket)

    try:
        # Send initial state
        if _optimizer:
            await websocket.send_json({
                "type": "state",
                "summary": _optimizer.get_draft_summary(),
            })

        while True:
            # Listen for client messages (e.g., requesting recalculation)
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "get_recommendations":
                if _optimizer and _optimizer.player_pool is not None:
                    recs = _optimizer.get_recommendations(top_n=25)
                    await websocket.send_json({
                        "type": "recommendations",
                        "recommendations": [asdict(r) for r in recs],
                    })

    except WebSocketDisconnect:
        _ws_clients.remove(websocket)


async def _broadcast(message: dict):
    """Send a message to all connected WebSocket clients."""
    disconnected = []
    for ws in _ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _ws_clients.remove(ws)
