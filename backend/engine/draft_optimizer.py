"""
Draft optimizer: the central orchestrator that ties together SGP,
Monte Carlo simulation, positional scarcity, and pitcher streaming
into a single recommendation engine.

This is the main interface used by the API layer. It maintains draft state,
recalculates after each pick, and produces ranked recommendations.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .config import LeagueSettings
from .sgp import (
    compute_player_sgp,
    compute_replacement_level,
    compute_value_above_replacement,
    DEFAULT_SGP_DENOMINATORS,
)
from .monte_carlo import (
    DraftState,
    evaluate_candidates,
    quick_rank,
    generate_draft_order,
    get_my_pick_positions,
    SimulationResult,
)
from .pitcher_streaming import compute_streaming_value, optimal_sp_count

logger = logging.getLogger(__name__)


@dataclass
class DraftPick:
    """Record of a single draft pick."""
    overall_pick: int
    team_idx: int
    player_id: str
    player_name: str
    round_num: int


@dataclass
class TeamRoster:
    """A team's drafted players."""
    team_idx: int
    player_ids: list = field(default_factory=list)

    def positions_filled(self, player_pool: pd.DataFrame) -> dict[str, int]:
        """Count how many of each position have been drafted."""
        counts = {}
        for pid in self.player_ids:
            if pid in player_pool.index:
                pos = player_pool.loc[pid].get("position", "UTIL")
                counts[pos] = counts.get(pos, 0) + 1
        return counts


@dataclass
class Recommendation:
    """A draft recommendation with supporting analysis."""
    rank: int
    player_id: str
    player_name: str
    position: str
    team: str
    var: float  # Value Above Replacement
    sgp_total: float
    rec_score: float
    availability_next_pick: float  # Probability available at next pick
    urgency: str  # "high", "medium", "low"
    reason: str  # Human-readable explanation


class DraftOptimizer:
    """
    Main draft optimization engine.

    Maintains the full state of a draft and produces recommendations
    that account for:
    - SGP-based player value (standings gain points)
    - Positional scarcity (value above replacement)
    - ADP-based availability (don't reach for players available later)
    - Positional need (fill roster holes)
    - Pitcher streaming value (extra SP benefit)
    """

    def __init__(
        self,
        settings: LeagueSettings,
        my_team_idx: int = 0,
        sgp_denominators: Optional[dict[str, float]] = None,
    ):
        self.settings = settings
        self.my_team_idx = my_team_idx
        self.sgp_denominators = sgp_denominators or DEFAULT_SGP_DENOMINATORS

        # Player data
        self.player_pool: Optional[pd.DataFrame] = None
        self._raw_hitters: Optional[pd.DataFrame] = None
        self._raw_pitchers: Optional[pd.DataFrame] = None

        # Draft state
        self.picks: list[DraftPick] = []
        self.rosters: dict[int, TeamRoster] = {
            i: TeamRoster(team_idx=i)
            for i in range(settings.num_teams)
        }
        self.draft_order = generate_draft_order(
            settings.num_teams, settings.num_rounds, settings.snake_draft
        )

        # Caches
        self._recommendations_cache: Optional[list[Recommendation]] = None
        self._cache_pick_num: int = -1

    def load_projections(
        self,
        player_pool: pd.DataFrame,
        hitters: Optional[pd.DataFrame] = None,
        pitchers: Optional[pd.DataFrame] = None,
    ):
        """
        Load the player pool and compute all derived values (SGP, VAR, etc.).

        Args:
            player_pool: Combined DataFrame of all players with projections.
            hitters: Optional separate hitter DataFrame (for streaming analysis).
            pitchers: Optional separate pitcher DataFrame (for streaming analysis).
        """
        self._raw_hitters = hitters
        self._raw_pitchers = pitchers

        # Compute SGP for all players
        pool = compute_player_sgp(player_pool, self.settings, self.sgp_denominators)

        # Compute replacement levels and VAR
        replacement_levels = compute_replacement_level(pool, self.settings)
        pool = compute_value_above_replacement(pool, self.settings, replacement_levels)

        self.player_pool = pool
        self._recommendations_cache = None

        logger.info(
            "Loaded %d players (%.0f hitters, %.0f pitchers)",
            len(pool),
            len(pool[pool.get("player_type", "") == "hitter"]) if "player_type" in pool.columns else 0,
            len(pool[pool.get("player_type", "") == "pitcher"]) if "player_type" in pool.columns else 0,
        )

    @property
    def current_pick(self) -> int:
        """Current overall pick number (0-indexed)."""
        return len(self.picks)

    @property
    def current_round(self) -> int:
        """Current round (1-indexed)."""
        return (self.current_pick // self.settings.num_teams) + 1

    @property
    def picking_team(self) -> int:
        """Team index that is currently on the clock."""
        if self.current_pick >= len(self.draft_order):
            return -1
        return self.draft_order[self.current_pick]

    @property
    def is_my_pick(self) -> bool:
        """Whether it's my team's turn to pick."""
        return self.picking_team == self.my_team_idx

    @property
    def available_player_ids(self) -> set:
        """Set of player IDs not yet drafted."""
        if self.player_pool is None:
            return set()
        drafted = set()
        for roster in self.rosters.values():
            drafted.update(roster.player_ids)
        return set(self.player_pool.index) - drafted

    @property
    def my_roster(self) -> list[str]:
        """Player IDs on my team."""
        return self.rosters[self.my_team_idx].player_ids

    @property
    def my_picks_remaining(self) -> list[int]:
        """Overall pick numbers remaining for my team."""
        all_my_picks = get_my_pick_positions(
            self.my_team_idx, self.settings.num_teams,
            self.settings.num_rounds, self.settings.snake_draft,
        )
        return [p for p in all_my_picks if p >= self.current_pick]

    def record_pick(self, team_idx: int, player_id: str, player_name: str = ""):
        """
        Record a draft pick and invalidate recommendation cache.

        Args:
            team_idx: Index of the team making the pick.
            player_id: ID of the player being drafted.
            player_name: Optional player name for display.
        """
        pick = DraftPick(
            overall_pick=self.current_pick,
            team_idx=team_idx,
            player_id=player_id,
            player_name=player_name or self._get_player_name(player_id),
            round_num=self.current_round,
        )
        self.picks.append(pick)
        self.rosters[team_idx].player_ids.append(player_id)
        self._recommendations_cache = None

        logger.info(
            "Pick %d (Rd %d): Team %d drafts %s",
            pick.overall_pick + 1, pick.round_num,
            team_idx, pick.player_name,
        )

    def undo_last_pick(self):
        """Undo the most recent draft pick."""
        if not self.picks:
            return
        pick = self.picks.pop()
        roster = self.rosters[pick.team_idx]
        if pick.player_id in roster.player_ids:
            roster.player_ids.remove(pick.player_id)
        self._recommendations_cache = None

    def get_recommendations(
        self,
        top_n: int = 25,
        use_monte_carlo: bool = False,
        n_simulations: int = 500,
    ) -> list[Recommendation]:
        """
        Get ranked player recommendations for the current pick.

        Args:
            top_n: Number of recommendations to return.
            use_monte_carlo: If True, run full Monte Carlo simulation
                             (slower but more accurate). If False, use
                             fast analytical ranking.
            n_simulations: Number of Monte Carlo simulations (if enabled).

        Returns:
            List of Recommendation objects, best first.
        """
        if self.player_pool is None:
            return []

        # Return cache if available for current pick
        if (self._recommendations_cache is not None
                and self._cache_pick_num == self.current_pick):
            return self._recommendations_cache[:top_n]

        draft_state = DraftState(
            my_team_idx=self.my_team_idx,
            current_pick=self.current_pick,
            my_roster=list(self.my_roster),
            all_rosters=[r.player_ids for r in self.rosters.values()],
            available_player_ids=self.available_player_ids,
            num_teams=self.settings.num_teams,
        )

        if use_monte_carlo:
            sim_results = evaluate_candidates(
                draft_state, self.player_pool, self.settings,
                top_n=top_n, n_simulations=n_simulations,
            )
            recommendations = [
                self._sim_result_to_recommendation(r, rank=i + 1)
                for i, r in enumerate(sim_results)
            ]
        else:
            ranked = quick_rank(self.player_pool, draft_state, self.settings)
            recommendations = []
            for i, (pid, row) in enumerate(ranked.head(top_n).iterrows()):
                recommendations.append(self._row_to_recommendation(row, rank=i + 1))

        self._recommendations_cache = recommendations
        self._cache_pick_num = self.current_pick

        return recommendations[:top_n]

    def get_streaming_analysis(self) -> dict:
        """
        Analyze the value of pitcher streaming for my team.

        Returns analysis of current SP situation and recommendation
        for how many more SPs to draft.
        """
        if self.player_pool is None:
            return {"error": "No projections loaded"}

        # Get my current SPs
        my_sps = []
        for pid in self.my_roster:
            if pid in self.player_pool.index:
                player = self.player_pool.loc[pid]
                if player.get("position") == "SP":
                    my_sps.append({
                        "era": player.get("ERA", 4.0),
                        "whip": player.get("WHIP", 1.2),
                        "ip": player.get("IP", 150),
                        "wins": player.get("W", 8),
                        "k": player.get("K", 120),
                        "starts": player.get("starts", player.get("GS", 28)),
                    })

        active_slots = self.settings.roster_slots.get("SP", 7)

        if len(my_sps) == 0:
            return {
                "current_sp_count": 0,
                "active_sp_slots": active_slots,
                "recommendation": "Draft starting pitchers",
                "streaming_benefit_sgp": 0,
            }

        analysis = compute_streaming_value(
            my_sps, active_slots, self.settings, n_simulations=2000,
        )

        # Estimate marginal hitter value
        available = self.player_pool[
            self.player_pool.index.isin(self.available_player_ids)
        ]
        hitters_avail = available[available.get("player_type", "") == "hitter"]
        marginal_hitter_sgp = 0.0
        if len(hitters_avail) > 0:
            sorted_h = hitters_avail.sort_values("VAR", ascending=False)
            # Take median of next few available hitters as the marginal value
            marginal_hitter_sgp = sorted_h.head(5)["VAR"].median()

        return {
            "current_sp_count": len(my_sps),
            "active_sp_slots": active_slots,
            "projected_era": round(analysis.projected_era, 2),
            "projected_whip": round(analysis.projected_whip, 3),
            "era_benefit": round(analysis.era_improvement_vs_fixed, 3),
            "whip_benefit": round(analysis.whip_improvement_vs_fixed, 4),
            "streaming_benefit_sgp": round(analysis.sgp_gained_from_streaming, 2),
            "recommendation": self._streaming_recommendation(
                len(my_sps), active_slots, analysis.sgp_gained_from_streaming,
                marginal_hitter_sgp,
            ),
        }

    def get_draft_summary(self) -> dict:
        """Get a summary of the current draft state."""
        my_team = self.player_pool.loc[
            self.player_pool.index.isin(self.my_roster)
        ] if self.player_pool is not None and self.my_roster else pd.DataFrame()

        # Category totals for my team (convert numpy types to native Python)
        cat_totals = {}
        for cat in self.settings.all_categories:
            if cat in my_team.columns:
                cat_totals[cat] = round(float(my_team[cat].sum()), 3)

        # Position counts
        pos_counts = {}
        for _, row in my_team.iterrows():
            pos = row.get("position", "UTIL")
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        return {
            "current_pick": int(self.current_pick + 1),
            "current_round": int(self.current_round),
            "picking_team": int(self.picking_team),
            "is_my_pick": bool(self.is_my_pick),
            "total_picks": int(len(self.draft_order)),
            "my_roster_size": int(len(self.my_roster)),
            "my_team_sgp": round(float(my_team["SGP_total"].sum()), 2) if "SGP_total" in my_team.columns else 0,
            "category_totals": cat_totals,
            "position_counts": pos_counts,
            "picks_remaining": int(len(self.my_picks_remaining)),
        }

    def _get_player_name(self, player_id: str) -> str:
        if self.player_pool is not None and player_id in self.player_pool.index:
            return self.player_pool.loc[player_id].get("name", str(player_id))
        return str(player_id)

    def _row_to_recommendation(self, row: pd.Series, rank: int) -> Recommendation:
        var_val = row.get("VAR", row.get("SGP_total", 0))
        avail = row.get("availability_next_pick", 0)

        if avail > 0.7:
            urgency = "low"
            reason = f"Likely available later ({avail:.0%} chance at next pick)"
        elif avail > 0.3:
            urgency = "medium"
            reason = f"Moderate risk of being taken ({1 - avail:.0%} chance gone)"
        else:
            urgency = "high"
            reason = f"High demand — likely gone by next pick"

        pos = row.get("position", "UTIL")
        if isinstance(pos, list):
            pos = pos[0] if pos else "UTIL"

        return Recommendation(
            rank=rank,
            player_id=str(row.get("player_id", row.name)),
            player_name=row.get("name", str(row.name)),
            position=pos,
            team=row.get("team", ""),
            var=round(float(var_val), 2),
            sgp_total=round(float(row.get("SGP_total", 0)), 2),
            rec_score=round(float(row.get("rec_score", var_val)), 2),
            availability_next_pick=round(float(avail), 2),
            urgency=urgency,
            reason=reason,
        )

    def _sim_result_to_recommendation(
        self, result: SimulationResult, rank: int
    ) -> Recommendation:
        if result.availability_at_next_pick > 0.7:
            urgency = "low"
            reason = f"Available later ({result.availability_at_next_pick:.0%}). Sim SGP: {result.mean_total_sgp:.1f}"
        elif result.availability_at_next_pick > 0.3:
            urgency = "medium"
            reason = f"Moderate risk. Sim SGP: {result.mean_total_sgp:.1f}"
        else:
            urgency = "high"
            reason = f"Take now! Sim SGP: {result.mean_total_sgp:.1f}"

        return Recommendation(
            rank=rank,
            player_id=result.player_id,
            player_name=result.player_name,
            position="",
            team="",
            var=round(result.marginal_sgp, 2),
            sgp_total=round(result.mean_total_sgp, 2),
            rec_score=round(result.overall_score, 2),
            availability_next_pick=round(result.availability_at_next_pick, 2),
            urgency=urgency,
            reason=reason,
        )

    def _streaming_recommendation(
        self, current_sp: int, active_slots: int,
        stream_sgp: float, marginal_hitter_sgp: float,
    ) -> str:
        extra = current_sp - active_slots
        if extra >= 3:
            return f"You have {extra} bench SPs. Streaming value is {stream_sgp:.1f} SGP. Consider shifting to hitters."
        elif extra >= 1:
            net = stream_sgp - marginal_hitter_sgp
            if net > 0:
                return f"Streaming {extra} extra SPs gains {stream_sgp:.1f} SGP. Draft another SP for more benefit."
            else:
                return f"Streaming benefit ({stream_sgp:.1f} SGP) < marginal hitter value ({marginal_hitter_sgp:.1f} SGP). Draft a hitter."
        else:
            return f"Fill your {active_slots - current_sp} remaining SP slots before considering streaming."
