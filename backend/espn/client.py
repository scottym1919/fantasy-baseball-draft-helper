"""
ESPN Fantasy Baseball API client.

Connects to ESPN's undocumented Fantasy API to:
  - Fetch league settings and rosters
  - Poll live draft picks in real time
  - Map ESPN player IDs to our internal player pool

Authentication uses espn_s2 and SWID cookies extracted from
a browser session logged into ESPN.

Base URL (as of 2024): https://lm-api-reads.fantasy.espn.com/apis/v3/games/flb/
"""

import httpx
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)

BASE_URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/flb"

# ESPN position ID to position name
ESPN_POSITION_ID_MAP = {
    0: "C", 1: "1B", 2: "2B", 3: "3B", 4: "SS",
    5: "LF", 6: "CF", 7: "RF",
    8: "DH", 9: "SP", 10: "RP",
    # Roster slot IDs
    12: "UTIL", 13: "P", 14: "SP", 15: "RP",
    16: "BN", 17: "IL",
}


@dataclass
class ESPNCredentials:
    """ESPN authentication credentials."""
    espn_s2: str
    swid: str  # Include the curly braces, e.g. "{XXXXXXXX-XXXX-...}"


@dataclass
class ESPNDraftPick:
    """A single draft pick from ESPN."""
    overall_pick: int
    round_num: int
    round_pick: int
    team_id: int
    player_id: int
    player_name: str = ""
    keeper: bool = False


@dataclass
class ESPNLeagueInfo:
    """Basic league information from ESPN."""
    league_id: int
    season: int
    name: str = ""
    num_teams: int = 12
    team_names: dict = field(default_factory=dict)  # team_id -> team_name
    roster_slots: dict = field(default_factory=dict)
    scoring_type: str = ""  # "ROTO", "H2H", etc.


class ESPNClient:
    """
    Async client for the ESPN Fantasy Baseball API.

    Usage:
        creds = ESPNCredentials(espn_s2="...", swid="{...}")
        client = ESPNClient(league_id=12345, season=2025, credentials=creds)

        league = await client.get_league_info()
        picks = await client.get_draft_picks()
        await client.poll_draft(callback=on_new_pick)
    """

    def __init__(
        self,
        league_id: int,
        season: int,
        credentials: Optional[ESPNCredentials] = None,
    ):
        self.league_id = league_id
        self.season = season
        self.credentials = credentials
        self._player_cache: dict[int, dict] = {}
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._http is None or self._http.is_closed:
            cookies = {}
            if self.credentials:
                cookies = {
                    "espn_s2": self.credentials.espn_s2,
                    "SWID": self.credentials.swid,
                }
            self._http = httpx.AsyncClient(
                cookies=cookies,
                timeout=30.0,
                headers={"Accept": "application/json"},
            )
        return self._http

    async def close(self):
        """Close the HTTP client."""
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    def _league_url(self) -> str:
        """Build the league-specific API URL."""
        return f"{BASE_URL}/seasons/{self.season}/segments/0/leagues/{self.league_id}"

    async def _fetch(self, params: Optional[dict] = None) -> dict:
        """Make an authenticated request to the ESPN API."""
        client = await self._get_client()
        url = self._league_url()
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def get_league_info(self) -> ESPNLeagueInfo:
        """Fetch basic league settings and team info."""
        data = await self._fetch(params={"view": "mSettings"})

        settings = data.get("settings", {})
        schedule = settings.get("scheduleSettings", {})
        roster = settings.get("rosterSettings", {})

        # Parse roster slot counts
        slot_counts = {}
        for slot in roster.get("lineupSlotCounts", {}):
            pos_name = ESPN_POSITION_ID_MAP.get(int(slot), f"UNK_{slot}")
            count = roster["lineupSlotCounts"][slot]
            if count > 0:
                slot_counts[pos_name] = count

        # Get team names
        team_names = {}
        for team in data.get("teams", []):
            team_names[team["id"]] = team.get("name", f"Team {team['id']}")

        scoring_type = settings.get("scoringSettings", {}).get("scoringType", "")

        return ESPNLeagueInfo(
            league_id=self.league_id,
            season=self.season,
            name=settings.get("name", ""),
            num_teams=len(data.get("teams", [])) or settings.get("size", 12),
            team_names=team_names,
            roster_slots=slot_counts,
            scoring_type=scoring_type,
        )

    async def get_draft_picks(self) -> list[ESPNDraftPick]:
        """Fetch all draft picks (completed so far)."""
        data = await self._fetch(params={"view": "mDraftDetail"})

        draft_detail = data.get("draftDetail", {})
        picks_raw = draft_detail.get("picks", [])

        picks = []
        for p in picks_raw:
            player_name = ""
            player_id = p.get("playerId", 0)
            if player_id and player_id in self._player_cache:
                player_name = self._player_cache[player_id].get("name", "")

            picks.append(ESPNDraftPick(
                overall_pick=p.get("overallPickNumber", 0),
                round_num=p.get("roundId", 0),
                round_pick=p.get("roundPickNumber", 0),
                team_id=p.get("teamId", 0),
                player_id=player_id,
                player_name=player_name,
                keeper=p.get("keeper", False),
            ))

        return picks

    async def get_players(self, limit: int = 500) -> dict[int, dict]:
        """
        Fetch the player universe from ESPN.

        Returns dict mapping ESPN player_id -> player info dict.
        """
        client = await self._get_client()

        # ESPN player endpoint with filters
        url = f"{BASE_URL}/seasons/{self.season}/players"
        headers = {
            "x-fantasy-filter": (
                '{"filterActive":{"value":true}}'
            ),
        }
        params = {
            "view": "kona_player_info",
            "scoringPeriodId": 0,
        }

        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        players = {}
        for p in data if isinstance(data, list) else data.get("players", []):
            player_data = p if "id" in p else p.get("player", {})
            pid = player_data.get("id", 0)
            if not pid:
                continue

            # Parse position eligibility
            positions = []
            for pos_id in player_data.get("eligibleSlots", []):
                pos_name = ESPN_POSITION_ID_MAP.get(pos_id)
                if pos_name and pos_name not in positions and pos_name not in ("BN", "IL"):
                    positions.append(pos_name)

            full_name = player_data.get("fullName", "")
            if not full_name:
                first = player_data.get("firstName", "")
                last = player_data.get("lastName", "")
                full_name = f"{first} {last}".strip()

            players[pid] = {
                "espn_id": pid,
                "name": full_name,
                "team": player_data.get("proTeamId", 0),
                "positions": positions,
                "default_position": ESPN_POSITION_ID_MAP.get(
                    player_data.get("defaultPositionId", 0), "UTIL"
                ),
                "ownership_pct": p.get("player", {}).get("ownership", {}).get("percentOwned", 0),
            }

        self._player_cache = players
        return players

    async def poll_draft(
        self,
        callback: Callable[[list[ESPNDraftPick], int], Awaitable[None]],
        poll_interval: float = 5.0,
        max_duration: float = 14400.0,  # 4 hours
    ):
        """
        Continuously poll for new draft picks during a live draft.

        Args:
            callback: Async function called with (new_picks, total_picks)
                      whenever new picks are detected.
            poll_interval: Seconds between polls.
            max_duration: Maximum polling duration in seconds.
        """
        known_picks = 0
        elapsed = 0.0

        logger.info("Starting draft poll (interval=%.1fs)", poll_interval)

        while elapsed < max_duration:
            try:
                picks = await self.get_draft_picks()
                total = len(picks)

                if total > known_picks:
                    new_picks = picks[known_picks:]
                    known_picks = total
                    logger.info("New picks detected: %d (total: %d)", len(new_picks), total)
                    await callback(new_picks, total)

            except httpx.HTTPStatusError as e:
                logger.warning("ESPN API error: %s", e)
            except httpx.ConnectError as e:
                logger.warning("ESPN connection error: %s", e)

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        logger.info("Draft polling ended after %.0f seconds", elapsed)

    async def get_draft_status(self) -> dict:
        """Get current draft status (in progress, completed, etc.)."""
        data = await self._fetch(params={"view": "mDraftDetail"})
        draft_detail = data.get("draftDetail", {})
        return {
            "drafted": draft_detail.get("drafted", False),
            "in_progress": draft_detail.get("inProgress", False),
            "picks_made": len(draft_detail.get("picks", [])),
        }


def match_espn_to_projections(
    espn_players: dict[int, dict],
    projection_pool: "pd.DataFrame",
) -> dict[int, str]:
    """
    Match ESPN player IDs to projection player IDs using name matching.

    Returns dict mapping espn_player_id -> projection_player_id.
    """
    import pandas as pd
    from difflib import SequenceMatcher

    matches = {}

    # Build name lookup from projections
    proj_names = {}
    for pid, row in projection_pool.iterrows():
        name = row.get("name", "")
        if name:
            proj_names[name.lower().strip()] = pid

    for espn_id, espn_info in espn_players.items():
        espn_name = espn_info.get("name", "").lower().strip()
        if not espn_name:
            continue

        # Exact match first
        if espn_name in proj_names:
            matches[espn_id] = proj_names[espn_name]
            continue

        # Fuzzy match
        best_score = 0.0
        best_pid = None
        for proj_name, pid in proj_names.items():
            score = SequenceMatcher(None, espn_name, proj_name).ratio()
            if score > best_score and score > 0.85:
                best_score = score
                best_pid = pid

        if best_pid:
            matches[espn_id] = best_pid

    return matches
