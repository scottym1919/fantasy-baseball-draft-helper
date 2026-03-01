"""
Standings Gain Points (SGP) engine for rotisserie scoring.

SGP is the gold standard for rotisserie valuation. It directly answers:
"How many standings points does one additional unit of a stat earn?"

SGP_denominator = average gap between adjacent teams in a category.
Gaining one SGP_denominator's worth of a stat ≈ 1 extra standings point.
"""

import numpy as np
import pandas as pd
from typing import Optional

from .config import (
    INVERSE_CATEGORIES, RATE_STATS, COUNTING_STATS,
    RATE_STAT_COMPONENTS, LeagueSettings,
)


# Historical SGP denominators (12-team 5x5 league defaults)
# These represent the average stat gap between consecutive teams in standings.
DEFAULT_SGP_DENOMINATORS = {
    "R": 22.0,
    "HR": 10.0,
    "RBI": 22.0,
    "SB": 10.0,
    "AVG": 0.004,
    "W": 4.0,
    "SV": 10.0,
    "K": 30.0,
    "ERA": 0.20,
    "WHIP": 0.020,
}

# Typical team totals for a 12-team league (used for rate stat SGP calculations)
TYPICAL_TEAM_AB = 5500
TYPICAL_TEAM_IP = 1300


def compute_sgp_denominators_from_history(
    historical_standings: pd.DataFrame,
    categories: list[str],
) -> dict[str, float]:
    """
    Compute SGP denominators from historical league standings data.

    For each category, sorts the 12 team totals and computes:
        SGP_denominator = (best - worst) / (num_teams - 1)

    Args:
        historical_standings: DataFrame with columns for each category,
                              rows are teams in a given year.
        categories: List of category names to compute.

    Returns:
        Dictionary mapping category name -> SGP denominator.
    """
    denominators = {}
    for cat in categories:
        if cat not in historical_standings.columns:
            continue
        values = historical_standings[cat].dropna().sort_values(ascending=False).values
        if len(values) < 2:
            continue
        denominators[cat] = (values[0] - values[-1]) / (len(values) - 1)

    return denominators


def compute_player_sgp(
    projections: pd.DataFrame,
    settings: LeagueSettings,
    sgp_denominators: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Convert player projections to SGP values for all categories.

    For counting stats: SGP = (projected - replacement_baseline) / denominator
    For rate stats (AVG, ERA, WHIP): volume-adjusted SGP calculation.

    Args:
        projections: DataFrame with player projections. Must have columns
                     for each stat category plus 'AB' and 'IP' for rate stats.
        settings: League configuration.
        sgp_denominators: Optional custom denominators. Uses defaults if None.

    Returns:
        DataFrame with original data plus SGP columns (e.g., 'SGP_R', 'SGP_HR').
    """
    denoms = sgp_denominators or DEFAULT_SGP_DENOMINATORS
    df = projections.copy()
    all_cats = settings.all_categories

    for cat in all_cats:
        if cat not in df.columns:
            df[f"SGP_{cat}"] = 0.0
            continue

        denom = denoms.get(cat, 1.0)
        if denom == 0:
            denom = 1.0

        if cat in COUNTING_STATS:
            # Counting stats: straightforward division
            df[f"SGP_{cat}"] = df[cat] / denom

        elif cat == "AVG":
            # AVG SGP depends on volume (AB)
            # SGP = (AVG - league_avg) * AB / (team_AB * denom)
            ab = df.get("AB", pd.Series(0, index=df.index))
            league_avg = df["AVG"].median()
            df[f"SGP_{cat}"] = (df["AVG"] - league_avg) * ab / (TYPICAL_TEAM_AB * denom)

        elif cat == "ERA":
            # ERA: lower is better, volume-weighted by IP
            ip = df.get("IP", pd.Series(0, index=df.index))
            league_era = df["ERA"].median()
            # Negative because lower ERA = more SGP
            df[f"SGP_{cat}"] = -1 * (df["ERA"] - league_era) * ip / (TYPICAL_TEAM_IP * denom)

        elif cat == "WHIP":
            # WHIP: lower is better, volume-weighted by IP
            ip = df.get("IP", pd.Series(0, index=df.index))
            league_whip = df["WHIP"].median()
            df[f"SGP_{cat}"] = -1 * (df["WHIP"] - league_whip) * ip / (TYPICAL_TEAM_IP * denom)

    # Total SGP across all categories
    sgp_cols = [f"SGP_{cat}" for cat in all_cats if f"SGP_{cat}" in df.columns]
    df["SGP_total"] = df[sgp_cols].sum(axis=1)

    return df


def compute_replacement_level(
    projections: pd.DataFrame,
    settings: LeagueSettings,
) -> dict[str, float]:
    """
    Compute the replacement-level SGP for each position.

    Replacement level = the SGP_total of the best freely available player
    at that position (i.e., the (N+1)-th best player, where N = number
    of starters at that position across the league).

    Args:
        projections: DataFrame with SGP_total and position columns.
        settings: League configuration.

    Returns:
        Dictionary mapping position -> replacement-level SGP_total.
    """
    replacement_levels = {}

    for pos, num_slots in settings.roster_slots.items():
        if pos == "BN":
            continue

        total_starters = settings.num_teams * num_slots

        # Filter players eligible at this position
        if "eligible_positions" in projections.columns:
            eligible = projections[
                projections["eligible_positions"].apply(
                    lambda positions: pos in positions if isinstance(positions, list) else pos == positions
                )
            ]
        elif "position" in projections.columns:
            eligible = projections[projections["position"] == pos]
        else:
            replacement_levels[pos] = 0.0
            continue

        if len(eligible) == 0:
            replacement_levels[pos] = 0.0
            continue

        # Sort by SGP_total descending, take the (N+1)-th player
        sorted_players = eligible.sort_values("SGP_total", ascending=False)
        replacement_idx = min(total_starters, len(sorted_players) - 1)
        replacement_levels[pos] = sorted_players.iloc[replacement_idx]["SGP_total"]

    return replacement_levels


def compute_value_above_replacement(
    projections: pd.DataFrame,
    settings: LeagueSettings,
    replacement_levels: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute Value Above Replacement (VAR) for each player.

    VAR = player's SGP_total - replacement_level(position)

    For multi-position eligible players, uses the most scarce position
    (highest replacement level) since that's where they provide most value.

    Args:
        projections: DataFrame with SGP_total computed.
        settings: League configuration.
        replacement_levels: Optional pre-computed levels. Computes if None.

    Returns:
        DataFrame with VAR column added.
    """
    if replacement_levels is None:
        replacement_levels = compute_replacement_level(projections, settings)

    df = projections.copy()

    def get_replacement_level(row):
        """Get replacement level for a player based on their position(s)."""
        if "eligible_positions" in df.columns and isinstance(row.get("eligible_positions"), list):
            positions = row["eligible_positions"]
        elif "position" in df.columns:
            positions = [row["position"]]
        else:
            return 0.0

        # Use the highest replacement level (most scarce position)
        levels = [replacement_levels.get(p, 0.0) for p in positions
                  if p in replacement_levels]
        return max(levels) if levels else 0.0

    df["replacement_level"] = df.apply(get_replacement_level, axis=1)
    df["VAR"] = df["SGP_total"] - df["replacement_level"]

    return df
