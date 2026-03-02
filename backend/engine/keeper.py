"""
Keeper assistant: helps decide which players to keep from existing rosters.

The keeper decision is a constrained optimization problem:
  - Each player belongs to a tier (based on prior draft round, salary, etc.)
  - Each tier has a maximum number of keepers allowed
  - Goal: select the combination of keepers that maximizes total team
    SGP value while respecting all tier constraints

Uses the SGP engine for valuation and combinatorial optimization
(ILP when scipy is available, brute-force enumeration for small
problems) to find the optimal keeper set.

Also accounts for:
  - Opportunity cost: keeping a player in tier T uses a keeper slot
    that could go to another player in that tier
  - Draft capital: each keeper "costs" a draft pick; the value of
    the pick you give up matters
  - League-wide keeper impact: if many top players are kept, the
    remaining draft pool changes, shifting replacement levels
"""

import logging
import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .config import LeagueSettings
from .sgp import (
    compute_player_sgp,
    compute_replacement_level,
    compute_value_above_replacement,
    DEFAULT_SGP_DENOMINATORS,
)

logger = logging.getLogger(__name__)


@dataclass
class KeeperTier:
    """Definition of a single keeper tier."""
    tier_id: int
    name: str  # e.g., "Tier 1 (Rounds 1-3)", "Tier 2 (Rounds 4-7)"
    max_keepers: int  # Maximum players that can be kept from this tier
    round_cost: Optional[int] = None  # Draft round this keeper costs (if applicable)
    description: str = ""


@dataclass
class KeeperCandidate:
    """A player eligible to be kept, with tier assignment and valuation."""
    player_id: str
    player_name: str
    position: str
    team: str
    tier_id: int
    tier_name: str
    sgp_total: float
    var: float  # Value Above Replacement
    keeper_value: float  # Net value = SGP gained minus draft pick cost
    draft_round_cost: Optional[int] = None  # Which round pick you give up
    prior_round: Optional[int] = None  # Round player was drafted in last year
    category_sgps: dict = field(default_factory=dict)


@dataclass
class KeeperRecommendation:
    """A recommended keeper selection with analysis."""
    kept_players: list[KeeperCandidate]
    total_sgp: float
    total_keeper_value: float  # Net value after accounting for pick costs
    tier_usage: dict  # tier_id -> number of keepers used
    category_totals: dict  # category -> total SGP from keepers
    explanation: str


class KeeperAssistant:
    """
    Keeper decision engine.

    Workflow:
    1. Configure tiers (how many keepers per tier)
    2. Load rosters for all teams (or just my team)
    3. Assign players to tiers
    4. Compute keeper values using SGP
    5. Find optimal keeper set via constrained optimization

    The engine considers:
    - Raw SGP value of each player
    - Cost of the draft pick surrendered
    - Positional scarcity of the remaining draft pool
    - What other teams are likely keeping (shifts replacement levels)
    """

    def __init__(
        self,
        settings: LeagueSettings,
        sgp_denominators: Optional[dict[str, float]] = None,
    ):
        self.settings = settings
        self.sgp_denominators = sgp_denominators or DEFAULT_SGP_DENOMINATORS
        self.tiers: list[KeeperTier] = []
        self.rosters: dict[int, list[dict]] = {}  # team_idx -> list of player dicts
        self.player_pool: Optional[pd.DataFrame] = None
        self._candidates: Optional[list[KeeperCandidate]] = None

    def configure_tiers(self, tiers: list[dict]):
        """
        Set up keeper tiers.

        Args:
            tiers: List of tier definitions, e.g.:
                [
                    {"name": "Tier 1 (Rds 1-3)", "max_keepers": 1, "round_cost": 1},
                    {"name": "Tier 2 (Rds 4-7)", "max_keepers": 2, "round_cost": 4},
                    {"name": "Tier 3 (Rds 8+)", "max_keepers": 3, "round_cost": 8},
                ]
        """
        self.tiers = []
        for i, t in enumerate(tiers):
            self.tiers.append(KeeperTier(
                tier_id=i,
                name=t.get("name", f"Tier {i + 1}"),
                max_keepers=t.get("max_keepers", 1),
                round_cost=t.get("round_cost"),
                description=t.get("description", ""),
            ))
        self._candidates = None
        logger.info("Configured %d keeper tiers", len(self.tiers))

    def load_roster(self, team_idx: int, players: list[dict]):
        """
        Load a team's roster for keeper evaluation.

        Args:
            team_idx: Team index.
            players: List of player dicts with at minimum:
                - player_id or name: identifier
                - tier: tier index (0-based) or tier name
                Optionally:
                - prior_round: which round they were drafted in last year
                - position: player position
        """
        self.rosters[team_idx] = players
        self._candidates = None
        logger.info("Loaded %d players for team %d", len(players), team_idx)

    def load_all_rosters(self, all_rosters: dict[int, list[dict]]):
        """Load rosters for all teams at once."""
        self.rosters = all_rosters
        self._candidates = None

    def load_projections(self, player_pool: pd.DataFrame):
        """
        Load projection data for valuation.

        This should be the same player pool used by the draft optimizer,
        with SGP values already computed. If SGP isn't computed yet,
        we'll compute it here.
        """
        if "SGP_total" not in player_pool.columns:
            pool = compute_player_sgp(player_pool, self.settings, self.sgp_denominators)
            replacement = compute_replacement_level(pool, self.settings)
            pool = compute_value_above_replacement(pool, self.settings, replacement)
            self.player_pool = pool
        else:
            self.player_pool = player_pool.copy()
        self._candidates = None

    def get_candidates(self, team_idx: int) -> list[KeeperCandidate]:
        """
        Get all keeper-eligible players for a team with valuations.

        Returns KeeperCandidate objects with SGP values and net keeper
        value (accounting for draft pick cost).
        """
        if team_idx not in self.rosters:
            return []
        if self.player_pool is None:
            raise ValueError("Load projections first")
        if not self.tiers:
            raise ValueError("Configure tiers first")

        roster = self.rosters[team_idx]
        pool = self.player_pool
        candidates = []

        for player_info in roster:
            # Resolve player in projection pool
            player_id = player_info.get("player_id", "")
            player_name = player_info.get("name", "")

            player_row = self._find_player(player_id, player_name)
            if player_row is None:
                logger.warning("Could not find player '%s' in projections", player_name or player_id)
                continue

            # Resolve tier
            tier = self._resolve_tier(player_info)
            if tier is None:
                logger.warning("Could not assign tier for player '%s'", player_name)
                continue

            # Compute keeper value
            sgp_total = float(player_row.get("SGP_total", 0))
            var = float(player_row.get("VAR", 0))

            # Draft pick cost: the SGP value of an average player drafted
            # in the round you're giving up
            pick_cost = self._estimate_pick_value(tier.round_cost)
            keeper_value = var - pick_cost

            # Category SGPs
            cat_sgps = {}
            for cat in self.settings.all_categories:
                col = f"SGP_{cat}"
                if col in player_row.index:
                    cat_sgps[cat] = round(float(player_row[col]), 3)

            pos = player_row.get("position", player_info.get("position", "UTIL"))
            if isinstance(pos, list):
                pos = pos[0] if pos else "UTIL"

            candidates.append(KeeperCandidate(
                player_id=str(player_row.get("player_id", player_row.name)),
                player_name=str(player_row.get("name", player_name)),
                position=pos,
                team=str(player_row.get("team", "")),
                tier_id=tier.tier_id,
                tier_name=tier.name,
                sgp_total=round(sgp_total, 2),
                var=round(var, 2),
                keeper_value=round(keeper_value, 2),
                draft_round_cost=tier.round_cost,
                prior_round=player_info.get("prior_round"),
                category_sgps=cat_sgps,
            ))

        # Sort by keeper_value descending within each tier
        candidates.sort(key=lambda c: c.keeper_value, reverse=True)
        self._candidates = candidates
        return candidates

    def find_optimal_keepers(
        self,
        team_idx: int,
        max_total_keepers: Optional[int] = None,
        other_teams_keepers: Optional[dict[int, list[str]]] = None,
    ) -> KeeperRecommendation:
        """
        Find the optimal set of keepers that maximizes total value
        while respecting tier constraints.

        This solves the constrained optimization problem:
            maximize Σ keeper_value(player)
            subject to:
                Σ kept(tier_t) <= max_keepers(tier_t)  for each tier t
                Σ kept(all) <= max_total_keepers        (optional overall cap)

        Uses ILP (Integer Linear Programming) via scipy when available,
        falls back to enumeration for small problem sizes.

        Args:
            team_idx: Team to optimize keepers for.
            max_total_keepers: Optional cap on total keepers across all tiers.
            other_teams_keepers: Optional dict of other teams' keeper lists
                                 (player_ids). Used to adjust replacement levels.

        Returns:
            KeeperRecommendation with the optimal set and analysis.
        """
        candidates = self.get_candidates(team_idx)
        if not candidates:
            return KeeperRecommendation(
                kept_players=[], total_sgp=0, total_keeper_value=0,
                tier_usage={}, category_totals={}, explanation="No eligible keeper candidates.",
            )

        # If other teams' keepers are known, adjust replacement levels
        if other_teams_keepers:
            self._adjust_for_league_keepers(candidates, other_teams_keepers)

        # Build tier -> candidate mapping
        tier_candidates: dict[int, list[KeeperCandidate]] = {}
        for c in candidates:
            tier_candidates.setdefault(c.tier_id, []).append(c)

        # Solve the optimization
        best_selection = self._solve_optimal_selection(
            candidates, tier_candidates, max_total_keepers,
        )

        # Build the recommendation
        total_sgp = sum(c.sgp_total for c in best_selection)
        total_value = sum(c.keeper_value for c in best_selection)

        tier_usage = {}
        for c in best_selection:
            tier_usage[c.tier_id] = tier_usage.get(c.tier_id, 0) + 1

        cat_totals = {}
        for c in best_selection:
            for cat, val in c.category_sgps.items():
                cat_totals[cat] = cat_totals.get(cat, 0) + val
        cat_totals = {k: round(v, 2) for k, v in cat_totals.items()}

        explanation = self._build_explanation(best_selection, tier_usage, candidates)

        return KeeperRecommendation(
            kept_players=best_selection,
            total_sgp=round(total_sgp, 2),
            total_keeper_value=round(total_value, 2),
            tier_usage=tier_usage,
            category_totals=cat_totals,
            explanation=explanation,
        )

    def compare_keeper_scenarios(
        self,
        team_idx: int,
        scenarios: list[list[str]],
    ) -> list[dict]:
        """
        Compare specific keeper scenarios side-by-side.

        Args:
            team_idx: Team index.
            scenarios: List of player_id lists, each representing a
                       possible keeper selection.

        Returns:
            List of scenario analyses with SGP totals and category breakdowns.
        """
        candidates = self.get_candidates(team_idx)
        candidate_map = {c.player_id: c for c in candidates}

        results = []
        for i, player_ids in enumerate(scenarios):
            kept = [candidate_map[pid] for pid in player_ids if pid in candidate_map]

            total_sgp = sum(c.sgp_total for c in kept)
            total_value = sum(c.keeper_value for c in kept)

            cat_totals = {}
            for c in kept:
                for cat, val in c.category_sgps.items():
                    cat_totals[cat] = cat_totals.get(cat, 0) + val

            tier_usage = {}
            for c in kept:
                tier_usage[c.tier_id] = tier_usage.get(c.tier_id, 0) + 1

            # Check tier constraint violations
            violations = []
            for tier in self.tiers:
                used = tier_usage.get(tier.tier_id, 0)
                if used > tier.max_keepers:
                    violations.append(
                        f"{tier.name}: keeping {used} but max is {tier.max_keepers}"
                    )

            results.append({
                "scenario": i + 1,
                "players": [{"name": c.player_name, "position": c.position,
                             "tier": c.tier_name, "keeper_value": c.keeper_value}
                            for c in kept],
                "total_sgp": round(total_sgp, 2),
                "total_keeper_value": round(total_value, 2),
                "category_totals": {k: round(v, 2) for k, v in cat_totals.items()},
                "tier_usage": tier_usage,
                "violations": violations,
                "valid": len(violations) == 0,
            })

        # Rank scenarios by total keeper value
        results.sort(key=lambda r: r["total_keeper_value"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    def get_keeper_impact_on_draft(
        self,
        all_teams_keepers: dict[int, list[str]],
    ) -> dict:
        """
        Analyze how all teams' keeper selections impact the draft pool.

        Shows which positions become scarcer and how replacement levels shift.
        """
        if self.player_pool is None:
            return {"error": "No projections loaded"}

        # Compute baseline replacement levels (no keepers)
        baseline_replacement = compute_replacement_level(self.player_pool, self.settings)

        # Remove all kept players from the pool
        all_kept = set()
        for player_ids in all_teams_keepers.values():
            all_kept.update(player_ids)

        remaining_pool = self.player_pool[~self.player_pool.index.isin(all_kept)]

        # Recompute replacement levels with reduced pool
        if "SGP_total" not in remaining_pool.columns:
            remaining_pool = compute_player_sgp(
                remaining_pool, self.settings, self.sgp_denominators
            )
        adjusted_replacement = compute_replacement_level(remaining_pool, self.settings)

        # Compute shifts
        position_impact = {}
        for pos in baseline_replacement:
            baseline = baseline_replacement[pos]
            adjusted = adjusted_replacement.get(pos, 0)
            shift = adjusted - baseline

            # Count how many keepers at this position
            kept_at_pos = 0
            for pid in all_kept:
                if pid in self.player_pool.index:
                    p_pos = self.player_pool.loc[pid].get("position", "")
                    if p_pos == pos:
                        kept_at_pos += 1

            position_impact[pos] = {
                "baseline_replacement_sgp": round(baseline, 2),
                "adjusted_replacement_sgp": round(adjusted, 2),
                "shift": round(shift, 2),
                "keepers_at_position": kept_at_pos,
                "scarcity_change": "more scarce" if shift > 0.5 else (
                    "less scarce" if shift < -0.5 else "minimal change"
                ),
            }

        return {
            "total_keepers": len(all_kept),
            "remaining_pool_size": len(remaining_pool),
            "position_impact": position_impact,
        }

    # ── Private methods ───────────────────────────────────────────────

    def _find_player(self, player_id: str, player_name: str) -> Optional[pd.Series]:
        """Find a player in the projection pool by ID or name."""
        pool = self.player_pool

        # Try exact ID match
        if player_id and player_id in pool.index:
            return pool.loc[player_id]

        # Try name match
        if player_name:
            name_lower = player_name.lower().strip()

            # Exact name match
            if "name" in pool.columns:
                exact = pool[pool["name"].str.lower().str.strip() == name_lower]
                if len(exact) > 0:
                    return exact.iloc[0]

                # Fuzzy contains match
                contains = pool[pool["name"].str.lower().str.contains(name_lower, na=False)]
                if len(contains) > 0:
                    return contains.iloc[0]

        return None

    def _resolve_tier(self, player_info: dict) -> Optional[KeeperTier]:
        """Resolve a player's tier from their info dict."""
        # Direct tier index
        if "tier" in player_info:
            tier_val = player_info["tier"]
            if isinstance(tier_val, int) and 0 <= tier_val < len(self.tiers):
                return self.tiers[tier_val]
            # Try matching by name
            if isinstance(tier_val, str):
                for t in self.tiers:
                    if t.name.lower() == tier_val.lower():
                        return t

        # Infer from prior_round if tier boundaries are defined by round_cost
        if "prior_round" in player_info and self.tiers:
            prior_round = player_info["prior_round"]
            # Assign to the tier whose round_cost is closest but not exceeding
            best_tier = None
            for t in sorted(self.tiers, key=lambda x: x.round_cost or 999, reverse=True):
                if t.round_cost is not None and prior_round >= t.round_cost:
                    best_tier = t
                    break
            if best_tier is None and self.tiers:
                best_tier = self.tiers[0]
            return best_tier

        # Default to first tier if only one exists
        if len(self.tiers) == 1:
            return self.tiers[0]

        return None

    def _estimate_pick_value(self, round_num: Optional[int]) -> float:
        """
        Estimate the SGP value of a draft pick in a given round.

        Uses a decay model: early picks are worth much more than late picks.
        The value represents what you could expect to draft at that position.
        """
        if round_num is None:
            return 0.0

        if self.player_pool is None or "VAR" not in self.player_pool.columns:
            return 0.0

        # Approximate: the Nth best available player, where N is the
        # overall pick number for that round
        overall_pick = (round_num - 1) * self.settings.num_teams + (self.settings.num_teams // 2)

        sorted_pool = self.player_pool.sort_values("VAR", ascending=False)
        if overall_pick < len(sorted_pool):
            return max(0, float(sorted_pool.iloc[overall_pick]["VAR"]))
        return 0.0

    def _solve_optimal_selection(
        self,
        candidates: list[KeeperCandidate],
        tier_candidates: dict[int, list[KeeperCandidate]],
        max_total: Optional[int],
    ) -> list[KeeperCandidate]:
        """
        Solve the keeper selection optimization problem.

        Tries ILP first (scipy.optimize.milp), falls back to enumeration.
        """
        # Try ILP approach
        try:
            return self._solve_ilp(candidates, tier_candidates, max_total)
        except Exception as e:
            logger.info("ILP solver unavailable or failed (%s), using enumeration", e)

        # Fallback: enumeration (fine for typical keeper sizes < 20 candidates)
        return self._solve_enumeration(candidates, tier_candidates, max_total)

    def _solve_ilp(
        self,
        candidates: list[KeeperCandidate],
        tier_candidates: dict[int, list[KeeperCandidate]],
        max_total: Optional[int],
    ) -> list[KeeperCandidate]:
        """Solve via Integer Linear Programming using scipy."""
        from scipy.optimize import linprog, LinearConstraint, milp, Bounds

        n = len(candidates)
        if n == 0:
            return []

        # Objective: maximize Σ keeper_value[i] * x[i]
        # milp minimizes, so negate the values
        c = np.array([-cand.keeper_value for cand in candidates])

        # Constraints: for each tier, Σ x[i] <= max_keepers[tier]
        A_ub = []
        b_ub = []
        for tier in self.tiers:
            row = np.zeros(n)
            for j, cand in enumerate(candidates):
                if cand.tier_id == tier.tier_id:
                    row[j] = 1.0
            A_ub.append(row)
            b_ub.append(tier.max_keepers)

        # Optional total keeper constraint
        if max_total is not None:
            row = np.ones(n)
            A_ub.append(row)
            b_ub.append(max_total)

        A_ub = np.array(A_ub) if A_ub else np.zeros((0, n))
        b_ub = np.array(b_ub) if b_ub else np.zeros(0)

        # Bounds: 0 <= x[i] <= 1 (binary)
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        integrality = np.ones(n)  # All variables are integer (binary)

        constraints = LinearConstraint(A_ub, ub=b_ub)

        result = milp(
            c=c, constraints=constraints,
            bounds=bounds, integrality=integrality,
        )

        if not result.success:
            raise RuntimeError(f"ILP solver failed: {result.message}")

        selected = [candidates[i] for i in range(n) if result.x[i] > 0.5]
        return selected

    def _solve_enumeration(
        self,
        candidates: list[KeeperCandidate],
        tier_candidates: dict[int, list[KeeperCandidate]],
        max_total: Optional[int],
    ) -> list[KeeperCandidate]:
        """
        Solve by enumerating all valid combinations per tier.

        Efficient for typical keeper sizes (5-15 candidates, 2-4 tiers).
        Uses cartesian product of per-tier combinations.
        """
        # For each tier, generate all valid subsets up to max_keepers
        tier_options: list[list[tuple[KeeperCandidate, ...]]] = []
        for tier in self.tiers:
            tier_cands = tier_candidates.get(tier.tier_id, [])
            options = [()]  # Empty set is always an option (keep nobody from this tier)
            for k in range(1, tier.max_keepers + 1):
                for combo in itertools.combinations(tier_cands, k):
                    options.append(combo)
            tier_options.append(options)

        # Enumerate all cross-tier combinations
        best_value = float("-inf")
        best_selection = []

        for combo in itertools.product(*tier_options):
            selection = []
            for tier_picks in combo:
                selection.extend(tier_picks)

            # Check total keeper cap
            if max_total is not None and len(selection) > max_total:
                continue

            # Only keep players with positive keeper value (don't keep
            # a player whose pick cost exceeds their value)
            # But allow user to override by including negative-value players
            total_value = sum(c.keeper_value for c in selection)
            if total_value > best_value:
                best_value = total_value
                best_selection = list(selection)

        return best_selection

    def _adjust_for_league_keepers(
        self,
        candidates: list[KeeperCandidate],
        other_teams_keepers: dict[int, list[str]],
    ):
        """
        Adjust keeper values based on what other teams are keeping.

        When many top players are kept, the remaining draft pool is weaker,
        which means your keepers are relatively more valuable. Conversely,
        if many players at a specific position are kept, that position
        becomes scarcer in the draft.
        """
        if self.player_pool is None:
            return

        # Collect all players being kept by other teams
        other_kept = set()
        for pids in other_teams_keepers.values():
            other_kept.update(pids)

        # Compute new replacement levels without kept players
        remaining = self.player_pool[~self.player_pool.index.isin(other_kept)]
        if "SGP_total" not in remaining.columns:
            return

        new_replacement = compute_replacement_level(remaining, self.settings)

        # Adjust candidate values based on new replacement levels
        for cand in candidates:
            new_repl = new_replacement.get(cand.position, 0)
            # Recompute VAR against the adjusted replacement level
            if cand.player_id in self.player_pool.index:
                player_sgp = float(self.player_pool.loc[cand.player_id].get("SGP_total", 0))
                cand.var = round(player_sgp - new_repl, 2)
                pick_cost = self._estimate_pick_value(cand.draft_round_cost)
                cand.keeper_value = round(cand.var - pick_cost, 2)

    def _build_explanation(
        self,
        selection: list[KeeperCandidate],
        tier_usage: dict,
        all_candidates: list[KeeperCandidate],
    ) -> str:
        """Build a human-readable explanation of the keeper recommendation."""
        if not selection:
            return "No players recommended for keeping. All candidates have negative keeper value (the draft pick you'd surrender is worth more)."

        lines = []
        lines.append(f"Keep {len(selection)} players for a total keeper value of {sum(c.keeper_value for c in selection):.1f} SGP above pick cost.")

        # Explain tier usage
        for tier in self.tiers:
            used = tier_usage.get(tier.tier_id, 0)
            tier_cands = [c for c in all_candidates if c.tier_id == tier.tier_id]
            if tier_cands:
                lines.append(f"  {tier.name}: keeping {used}/{tier.max_keepers} (from {len(tier_cands)} eligible)")

        # Highlight best value keepers
        best = sorted(selection, key=lambda c: c.keeper_value, reverse=True)
        if best:
            top = best[0]
            lines.append(f"Best value: {top.player_name} ({top.position}) at {top.keeper_value:.1f} SGP above pick cost.")

        # Note any high-value players NOT kept
        kept_ids = {c.player_id for c in selection}
        skipped = [c for c in all_candidates if c.player_id not in kept_ids and c.keeper_value > 0]
        if skipped:
            top_skipped = sorted(skipped, key=lambda c: c.keeper_value, reverse=True)[:3]
            skip_names = ", ".join(f"{c.player_name} ({c.keeper_value:.1f})" for c in top_skipped)
            lines.append(f"Notable players not kept (tier constraints): {skip_names}")

        return " ".join(lines)
