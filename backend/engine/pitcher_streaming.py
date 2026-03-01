"""
Pitcher streaming / rotation value calculator.

Models the statistical benefit of rostering extra starting pitchers
to rotate through lineup slots. The key insight: if you draft 10 SPs
for 7 slots, you can bench pitchers in bad matchups, improving your
rate stats (ERA, WHIP) while maintaining counting stats (W, K).

This module quantifies that benefit using order statistics and
Monte Carlo simulation of weekly start decisions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .config import LeagueSettings


@dataclass
class StreamingAnalysis:
    """Results of pitcher streaming value calculation."""

    num_sps_rostered: int
    active_sp_slots: int
    projected_ip: float
    projected_era: float
    projected_whip: float
    projected_wins: float
    projected_k: float
    era_improvement_vs_fixed: float  # ERA benefit from streaming
    whip_improvement_vs_fixed: float
    wins_improvement_vs_fixed: float
    k_improvement_vs_fixed: float
    sgp_gained_from_streaming: float


def compute_streaming_value(
    pitcher_projections: list[dict],
    active_slots: int,
    settings: LeagueSettings,
    n_simulations: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> StreamingAnalysis:
    """
    Compute the value of streaming pitchers vs using a fixed rotation.

    Simulates a full season of weekly start decisions. Each week,
    the best N starts (out of available M pitchers) are selected based
    on projected matchup quality.

    Args:
        pitcher_projections: List of dicts with keys:
            - era: float (projected ERA)
            - whip: float (projected WHIP)
            - ip: float (projected total IP)
            - wins: float (projected wins)
            - k: float (projected strikeouts)
            - starts: int (projected number of starts)
        active_slots: Number of active SP roster slots.
        settings: League settings.
        n_simulations: Number of season simulations.
        rng: Random number generator.

    Returns:
        StreamingAnalysis with projected stats and streaming benefit.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_pitchers = len(pitcher_projections)
    weeks = settings.season_weeks

    if n_pitchers == 0:
        return StreamingAnalysis(
            num_sps_rostered=0, active_sp_slots=active_slots,
            projected_ip=0, projected_era=0, projected_whip=0,
            projected_wins=0, projected_k=0,
            era_improvement_vs_fixed=0, whip_improvement_vs_fixed=0,
            wins_improvement_vs_fixed=0, k_improvement_vs_fixed=0,
            sgp_gained_from_streaming=0,
        )

    # Convert projections to per-start averages
    per_start = []
    for p in pitcher_projections:
        starts = max(p.get("starts", 30), 1)
        per_start.append({
            "ip_per_start": p["ip"] / starts,
            "era": p["era"],
            "whip": p["whip"],
            "er_per_start": (p["era"] / 9) * (p["ip"] / starts),
            "bbh_per_start": p["whip"] * (p["ip"] / starts),
            "w_per_start": p["wins"] / starts,
            "k_per_start": p["k"] / starts,
            "starts_per_week": starts / weeks,
        })

    # Matchup variance: ERA varies ~20% from start to start due to
    # opponent quality, park factors, and general variance
    matchup_variance_era = 0.20
    matchup_variance_whip = 0.15

    # Simulate seasons for streaming scenario (pick best N of M each week)
    stream_ip = np.zeros(n_simulations)
    stream_er = np.zeros(n_simulations)
    stream_bbh = np.zeros(n_simulations)
    stream_w = np.zeros(n_simulations)
    stream_k = np.zeros(n_simulations)

    # Simulate seasons for fixed scenario (use top N pitchers for all starts)
    fixed_ip = np.zeros(n_simulations)
    fixed_er = np.zeros(n_simulations)
    fixed_bbh = np.zeros(n_simulations)
    fixed_w = np.zeros(n_simulations)
    fixed_k = np.zeros(n_simulations)

    # Determine the "fixed" top pitchers by projected ERA
    sorted_by_era = sorted(range(n_pitchers), key=lambda i: per_start[i]["era"])
    fixed_roster = set(sorted_by_era[:active_slots])

    for sim in range(n_simulations):
        for week in range(weeks):
            # Each pitcher has a probability of starting this week
            available_starts = []
            for i, ps in enumerate(per_start):
                # Probability of having a start this week
                if rng.random() < ps["starts_per_week"]:
                    # Generate matchup-adjusted ERA for this start
                    start_era = max(0.5, rng.normal(ps["era"], ps["era"] * matchup_variance_era))
                    start_whip = max(0.5, rng.normal(ps["whip"], ps["whip"] * matchup_variance_whip))
                    start_ip = max(3.0, rng.normal(ps["ip_per_start"], 1.0))
                    start_er = (start_era / 9) * start_ip
                    start_bbh = start_whip * start_ip
                    start_w = 1.0 if rng.random() < ps["w_per_start"] else 0.0
                    start_k = max(0, rng.normal(ps["k_per_start"], 2.0))

                    available_starts.append({
                        "pitcher_idx": i,
                        "ip": start_ip,
                        "er": start_er,
                        "bbh": start_bbh,
                        "w": start_w,
                        "k": start_k,
                        "projected_era": start_era,
                    })

            # STREAMING: pick the best N starts by projected ERA
            available_starts.sort(key=lambda s: s["projected_era"])
            selected_starts = available_starts[:active_slots]

            for s in selected_starts:
                stream_ip[sim] += s["ip"]
                stream_er[sim] += s["er"]
                stream_bbh[sim] += s["bbh"]
                stream_w[sim] += s["w"]
                stream_k[sim] += s["k"]

            # FIXED: only use starts from the fixed roster
            fixed_starts = [s for s in available_starts if s["pitcher_idx"] in fixed_roster]
            for s in fixed_starts:
                fixed_ip[sim] += s["ip"]
                fixed_er[sim] += s["er"]
                fixed_bbh[sim] += s["bbh"]
                fixed_w[sim] += s["w"]
                fixed_k[sim] += s["k"]

    # Compute average stats
    def safe_rate(num, denom, multiplier=1.0):
        d = denom.mean()
        if d == 0:
            return 0.0
        return multiplier * num.mean() / d

    stream_era = safe_rate(stream_er, stream_ip, 9.0)
    stream_whip = safe_rate(stream_bbh, stream_ip)
    fixed_era = safe_rate(fixed_er, fixed_ip, 9.0)
    fixed_whip = safe_rate(fixed_bbh, fixed_ip)

    era_improvement = fixed_era - stream_era  # Positive = streaming is better
    whip_improvement = fixed_whip - stream_whip
    wins_improvement = stream_w.mean() - fixed_w.mean()
    k_improvement = stream_k.mean() - fixed_k.mean()

    # Convert improvements to SGP
    from .sgp import DEFAULT_SGP_DENOMINATORS
    sgp_gained = (
        era_improvement / DEFAULT_SGP_DENOMINATORS.get("ERA", 0.20) +
        whip_improvement / DEFAULT_SGP_DENOMINATORS.get("WHIP", 0.020) +
        wins_improvement / DEFAULT_SGP_DENOMINATORS.get("W", 4.0) +
        k_improvement / DEFAULT_SGP_DENOMINATORS.get("K", 30.0)
    )

    return StreamingAnalysis(
        num_sps_rostered=n_pitchers,
        active_sp_slots=active_slots,
        projected_ip=stream_ip.mean(),
        projected_era=stream_era,
        projected_whip=stream_whip,
        projected_wins=stream_w.mean(),
        projected_k=stream_k.mean(),
        era_improvement_vs_fixed=era_improvement,
        whip_improvement_vs_fixed=whip_improvement,
        wins_improvement_vs_fixed=wins_improvement,
        k_improvement_vs_fixed=k_improvement,
        sgp_gained_from_streaming=sgp_gained,
    )


def optimal_sp_count(
    all_sp_projections: list[dict],
    hitter_marginal_sgp: float,
    active_sp_slots: int,
    settings: LeagueSettings,
    max_sp: int = 12,
) -> dict:
    """
    Find the optimal number of SPs to roster by comparing streaming
    benefit against the cost of losing hitter production.

    Args:
        all_sp_projections: All available SP projections, sorted by quality.
        hitter_marginal_sgp: SGP of the marginal hitter you'd draft instead
                             of an extra SP.
        active_sp_slots: Number of active SP roster slots.
        settings: League settings.
        max_sp: Maximum number of SPs to consider.

    Returns:
        Dict with 'optimal_count', 'analysis_by_count' keys.
    """
    results = {}
    best_net_sgp = float("-inf")
    best_count = active_sp_slots

    for n_sp in range(active_sp_slots, min(max_sp + 1, len(all_sp_projections) + 1)):
        pitchers = all_sp_projections[:n_sp]
        analysis = compute_streaming_value(
            pitchers, active_sp_slots, settings, n_simulations=2000
        )

        # Cost: extra SPs beyond active slots cost a hitter each
        extra_sp = max(0, n_sp - active_sp_slots)
        hitter_cost = extra_sp * hitter_marginal_sgp

        net_sgp = analysis.sgp_gained_from_streaming - hitter_cost

        results[n_sp] = {
            "streaming_sgp": analysis.sgp_gained_from_streaming,
            "hitter_cost_sgp": hitter_cost,
            "net_sgp": net_sgp,
            "projected_era": analysis.projected_era,
            "projected_whip": analysis.projected_whip,
        }

        if net_sgp > best_net_sgp:
            best_net_sgp = net_sgp
            best_count = n_sp

    return {
        "optimal_count": best_count,
        "analysis_by_count": results,
    }
