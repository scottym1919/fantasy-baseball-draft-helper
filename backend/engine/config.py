"""League configuration and category definitions for rotisserie scoring."""

from dataclasses import dataclass, field


# Standard 5x5 rotisserie categories
HITTING_CATEGORIES = ["R", "HR", "RBI", "SB", "AVG"]
PITCHING_CATEGORIES = ["W", "SV", "K", "ERA", "WHIP"]
ALL_CATEGORIES = HITTING_CATEGORIES + PITCHING_CATEGORIES

# Categories where lower is better
INVERSE_CATEGORIES = {"ERA", "WHIP"}

# Rate stats vs counting stats (affects how we aggregate)
RATE_STATS = {"AVG", "ERA", "WHIP"}
COUNTING_STATS = {"R", "HR", "RBI", "SB", "W", "SV", "K"}

# Weighted components for rate stat calculation
# AVG = H / AB, ERA = ER * 9 / IP, WHIP = (BB + H) / IP
RATE_STAT_COMPONENTS = {
    "AVG": {"numerator": "H", "denominator": "AB"},
    "ERA": {"numerator": "ER", "denominator": "IP", "multiplier": 9},
    "WHIP": {"numerator": "BB_plus_H", "denominator": "IP"},
}

# Eligible positions and their roster slots
POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "UTIL", "SP", "RP"]

# Position eligibility groupings (a player may qualify at multiple positions)
POSITION_GROUPS = {
    "C": ["C"],
    "1B": ["1B"],
    "2B": ["2B"],
    "3B": ["3B"],
    "SS": ["SS"],
    "OF": ["OF", "LF", "CF", "RF"],
    "UTIL": ["C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH"],
    "SP": ["SP"],
    "RP": ["RP"],
}


@dataclass
class LeagueSettings:
    """Configurable league settings."""

    num_teams: int = 12
    roster_slots: dict = field(default_factory=lambda: {
        "C": 1,
        "1B": 1,
        "2B": 1,
        "3B": 1,
        "SS": 1,
        "OF": 3,
        "UTIL": 2,
        "SP": 7,
        "RP": 3,
        "BN": 5,
    })
    hitting_categories: list = field(default_factory=lambda: list(HITTING_CATEGORIES))
    pitching_categories: list = field(default_factory=lambda: list(PITCHING_CATEGORIES))
    num_rounds: int = 23
    snake_draft: bool = True

    # Pitcher streaming settings
    max_starts_per_week: int = 12  # ESPN typical weekly start limit
    games_per_week: float = 6.5  # Average games per week in a season
    season_weeks: int = 23  # Approximate weeks in fantasy season

    @property
    def all_categories(self) -> list:
        return self.hitting_categories + self.pitching_categories

    @property
    def total_roster_size(self) -> int:
        return sum(self.roster_slots.values())

    @property
    def active_hitter_slots(self) -> int:
        return sum(v for k, v in self.roster_slots.items()
                   if k not in ("SP", "RP", "BN"))

    @property
    def active_pitcher_slots(self) -> int:
        return self.roster_slots.get("SP", 0) + self.roster_slots.get("RP", 0)

    @property
    def total_draftable_players(self) -> int:
        return self.num_teams * self.num_rounds
