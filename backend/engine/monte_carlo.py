"""
Monte Carlo simulation engine for draft optimization.

Simulates thousands of draft scenarios to determine optimal pick decisions.
Uses opponent modeling (noisy ADP-based selection) to estimate:
  - Probability each player is available at each pick
  - Expected total standings points for different draft strategies
  - Optimal player to draft at each position

The key insight: each draft pick changes the value landscape for all
remaining picks. Monte Carlo lets us estimate these cascading effects.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from .config import LeagueSettings, INVERSE_CATEGORIES


@dataclass
class DraftState:
    """Current state of a draft in progress."""

    my_team_idx: int  # 0-indexed position in draft order
    current_pick: int  # Overall pick number (0-indexed)
    my_roster: list  # Player IDs already drafted by my team
    all_rosters: list  # List of lists, one per team
    available_player_ids: set  # Set of remaining player IDs
    num_teams: int = 12


@dataclass
class SimulationResult:
    """Results from a batch of Monte Carlo simulations."""

    player_id: str
    player_name: str
    mean_total_sgp: float
    std_total_sgp: float
    availability_at_next_pick: float  # Probability still available at next pick
    marginal_sgp: float  # SGP gained by drafting this player
    positional_need_bonus: float  # Extra value from filling a scarce position
    overall_score: float  # Combined recommendation score


def generate_draft_order(num_teams: int, num_rounds: int, snake: bool = True) -> list[int]:
    """
    Generate the complete draft order for a snake or linear draft.

    Returns list of team indices (0-indexed) for each overall pick.
    """
    order = []
    for round_num in range(num_rounds):
        round_order = list(range(num_teams))
        if snake and round_num % 2 == 1:
            round_order.reverse()
        order.extend(round_order)
    return order


def get_my_pick_positions(
    my_team_idx: int, num_teams: int, num_rounds: int, snake: bool = True
) -> list[int]:
    """Get all overall pick numbers (0-indexed) for a given team."""
    draft_order = generate_draft_order(num_teams, num_rounds, snake)
    return [i for i, team in enumerate(draft_order) if team == my_team_idx]


def opponent_pick_softmax(
    available_players: pd.DataFrame,
    temperature: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """
    Model an opponent's pick using softmax (Boltzmann) selection on ADP.

    Players with lower ADP rank (better) are more likely to be selected.
    Temperature controls randomness:
      - Low temp (0.1): almost always picks best ADP
      - High temp (1.0+): more random/unpredictable opponents

    Args:
        available_players: DataFrame with 'adp_rank' column (1=best).
        temperature: Controls opponent rationality.
        rng: Random number generator.

    Returns:
        Player ID of the selected player.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(available_players) == 0:
        return None

    # Compute selection probabilities via softmax on negative ADP rank
    ranks = available_players["adp_rank"].values.astype(float)
    # Shift ranks to prevent overflow: use rank among available players
    relative_ranks = np.argsort(np.argsort(ranks)).astype(float) + 1

    logits = -relative_ranks / temperature
    logits -= logits.max()  # Numerical stability
    probs = np.exp(logits)
    probs /= probs.sum()

    idx = rng.choice(len(available_players), p=probs)
    return available_players.index[idx]


def simulate_remaining_draft(
    draft_state: DraftState,
    player_pool: pd.DataFrame,
    settings: LeagueSettings,
    candidate_player_id: str,
    n_simulations: int = 1000,
    opponent_temperature: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Simulate the rest of the draft after picking a specific player.

    For each simulation:
    1. Draft the candidate player for my team
    2. Simulate all remaining picks (opponents use softmax ADP model)
    3. For my remaining picks, use greedy best-VAR-available
    4. Compute my team's total SGP

    Args:
        draft_state: Current state of the draft.
        player_pool: Full player pool with SGP values and ADP.
        settings: League settings.
        candidate_player_id: Player being evaluated.
        n_simulations: Number of draft simulations to run.
        opponent_temperature: Opponent rationality parameter.
        rng: Random number generator.

    Returns:
        Mean total SGP across all simulations.
    """
    if rng is None:
        rng = np.random.default_rng()

    draft_order = generate_draft_order(settings.num_teams, settings.num_rounds, settings.snake_draft)
    total_sgps = np.zeros(n_simulations)

    # Pre-sort player pool by VAR for greedy selection
    pool_sorted = player_pool.sort_values("VAR", ascending=False)

    for sim in range(n_simulations):
        # Initialize available players
        available = set(draft_state.available_player_ids)
        my_roster_ids = list(draft_state.my_roster)

        # Draft the candidate
        available.discard(candidate_player_id)
        my_roster_ids.append(candidate_player_id)

        # Simulate remaining picks
        for pick_num in range(draft_state.current_pick + 1, len(draft_order)):
            if not available:
                break

            team_idx = draft_order[pick_num]
            available_df = pool_sorted[pool_sorted.index.isin(available)]

            if len(available_df) == 0:
                break

            if team_idx == draft_state.my_team_idx:
                # My pick: greedy best VAR available
                pick_id = available_df.index[0]
                my_roster_ids.append(pick_id)
            else:
                # Opponent pick: softmax on ADP
                if "adp_rank" in available_df.columns:
                    pick_id = opponent_pick_softmax(available_df, opponent_temperature, rng)
                else:
                    pick_id = available_df.index[0]

            available.discard(pick_id)

        # Compute my team's total SGP
        my_team = player_pool.loc[player_pool.index.isin(my_roster_ids)]
        total_sgps[sim] = my_team["SGP_total"].sum()

    return total_sgps.mean()


def evaluate_candidates(
    draft_state: DraftState,
    player_pool: pd.DataFrame,
    settings: LeagueSettings,
    top_n: int = 20,
    n_simulations: int = 500,
    opponent_temperature: float = 0.5,
) -> list[SimulationResult]:
    """
    Evaluate top candidate players for the current pick using Monte Carlo.

    For each of the top-N candidates (by VAR), simulates the rest of
    the draft and estimates the total team SGP.

    Args:
        draft_state: Current draft state.
        player_pool: Player pool with SGP/VAR computed.
        settings: League settings.
        top_n: Number of candidates to evaluate.
        n_simulations: Simulations per candidate.
        opponent_temperature: Opponent model parameter.

    Returns:
        List of SimulationResult sorted by overall score (best first).
    """
    rng = np.random.default_rng()

    # Filter to available players and get top candidates by VAR
    available = player_pool[player_pool.index.isin(draft_state.available_player_ids)]
    candidates = available.nlargest(top_n, "VAR")

    # Compute next pick position for availability estimates
    my_picks = get_my_pick_positions(
        draft_state.my_team_idx, settings.num_teams,
        settings.num_rounds, settings.snake_draft
    )
    future_picks = [p for p in my_picks if p > draft_state.current_pick]
    next_pick = future_picks[0] if future_picks else None

    results = []
    for player_id, player in candidates.iterrows():
        # Run Monte Carlo simulation for this candidate
        mean_sgp = simulate_remaining_draft(
            draft_state, player_pool, settings,
            player_id, n_simulations, opponent_temperature, rng,
        )

        # Estimate availability at next pick
        availability = 0.0
        if next_pick is not None and "adp" in player_pool.columns:
            adp = player.get("adp", draft_state.current_pick)
            adp_std = player.get("adp_std", max(adp * 0.15, 3.0))
            # P(available at next pick) = P(drafted_at > next_pick)
            picks_until_next = next_pick - draft_state.current_pick
            if adp_std > 0:
                z = (next_pick - adp) / adp_std
                # Use logistic approximation for sharper transition
                availability = 1.0 / (1.0 + np.exp(0.8 * z))
            else:
                availability = 1.0 if next_pick < adp else 0.0

        # Positional need bonus
        pos_bonus = _compute_positional_need(
            player, draft_state.my_roster, player_pool, settings
        )

        # Overall score combines simulation result with availability discount
        # If a player will almost certainly be available later, lower priority
        urgency = 1.0 - (availability * 0.5)  # Scale urgency 0.5-1.0
        overall = mean_sgp * urgency + pos_bonus

        results.append(SimulationResult(
            player_id=str(player_id),
            player_name=player.get("name", str(player_id)),
            mean_total_sgp=mean_sgp,
            std_total_sgp=0.0,  # Could compute from sim distribution
            availability_at_next_pick=availability,
            marginal_sgp=player.get("VAR", 0.0),
            positional_need_bonus=pos_bonus,
            overall_score=overall,
        ))

    results.sort(key=lambda r: r.overall_score, reverse=True)
    return results


def _compute_positional_need(
    player: pd.Series,
    my_roster_ids: list,
    player_pool: pd.DataFrame,
    settings: LeagueSettings,
) -> float:
    """
    Compute bonus value for filling a positional need.

    Returns a positive bonus if the player fills a position where
    we have remaining roster slots and the position is scarce.
    """
    # Determine player's position(s)
    if "eligible_positions" in player.index:
        positions = player["eligible_positions"]
        if isinstance(positions, str):
            positions = [positions]
    elif "position" in player.index:
        positions = [player["position"]]
    else:
        return 0.0

    if not isinstance(positions, list):
        positions = [positions]

    # Count how many of each position we already have
    my_team = player_pool.loc[player_pool.index.isin(my_roster_ids)]
    position_counts = {}
    for _, p in my_team.iterrows():
        pos = p.get("position", "UTIL")
        position_counts[pos] = position_counts.get(pos, 0) + 1

    bonus = 0.0
    for pos in positions:
        if pos == "BN":
            continue
        needed = settings.roster_slots.get(pos, 0)
        have = position_counts.get(pos, 0)
        if have < needed:
            # Unfilled position: bonus based on fill percentage remaining.
            # A position with 1 slot unfilled is more urgent than one with
            # 7 slots unfilled (early in draft, all multi-slot positions
            # are "unfilled" but that's not urgent).
            fill_pct = have / max(needed, 1)
            # Single-slot positions (C, SS) get higher urgency when empty
            if needed <= 2:
                pos_bonus = 2.0 * (1 - fill_pct)
            else:
                pos_bonus = 1.0 * (1 - fill_pct)
            bonus = max(bonus, pos_bonus)

    return bonus


def quick_rank(
    player_pool: pd.DataFrame,
    draft_state: DraftState,
    settings: LeagueSettings,
) -> pd.DataFrame:
    """
    Fast ranking without full Monte Carlo (for initial display / fast mode).

    Uses VAR + ADP availability adjustment + positional need.
    Much faster than full simulation — suitable for real-time updates.

    Returns DataFrame with ranking columns added, sorted by recommendation.
    """
    available = player_pool[player_pool.index.isin(draft_state.available_player_ids)].copy()

    if len(available) == 0:
        return available

    # Get next pick position
    my_picks = get_my_pick_positions(
        draft_state.my_team_idx, settings.num_teams,
        settings.num_rounds, settings.snake_draft
    )
    future_picks = [p for p in my_picks if p > draft_state.current_pick]
    next_pick = future_picks[0] if future_picks else draft_state.current_pick + settings.num_teams

    # Availability score: discount players likely to still be available
    if "adp" in available.columns:
        adp = available["adp"].values
        adp_std = available.get("adp_std", pd.Series(
            np.maximum(adp * 0.15, 3.0), index=available.index
        )).values
        z = (next_pick - adp) / np.maximum(adp_std, 0.1)
        availability = 1.0 / (1.0 + np.exp(0.8 * z))
        urgency = 1.0 - (availability * 0.5)
    else:
        urgency = np.ones(len(available))
        availability = np.zeros(len(available))

    available["availability_next_pick"] = availability
    available["urgency"] = urgency

    # Positional need bonus for each player
    pos_bonuses = []
    for _, player in available.iterrows():
        bonus = _compute_positional_need(
            player, draft_state.my_roster, player_pool, settings
        )
        pos_bonuses.append(bonus)
    available["pos_need_bonus"] = pos_bonuses

    # Combined recommendation score
    var_col = "VAR" if "VAR" in available.columns else "SGP_total"
    available["rec_score"] = (
        available[var_col] * available["urgency"] + available["pos_need_bonus"]
    )

    return available.sort_values("rec_score", ascending=False)
