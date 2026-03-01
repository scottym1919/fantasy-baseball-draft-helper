"""
Projection data loader supporting FanGraphs CSV exports.

Handles Steamer, ZiPS, ATC, THE BAT, and custom projection formats.
Normalizes column names and computes derived fields needed for the
SGP engine.
"""

import pandas as pd
import io
from enum import Enum
from typing import Optional


class ProjectionSystem(str, Enum):
    STEAMER = "steamer"
    ZIPS = "zips"
    ATC = "atc"
    THE_BAT = "thebat"
    CUSTOM = "custom"


class PlayerType(str, Enum):
    HITTER = "hitter"
    PITCHER = "pitcher"


# Column name mappings: FanGraphs standard -> internal names
HITTER_COLUMN_MAP = {
    "Name": "name",
    "Team": "team",
    "G": "G",
    "PA": "PA",
    "AB": "AB",
    "H": "H",
    "1B": "1B",
    "2B": "2B",
    "3B": "3B",
    "HR": "HR",
    "R": "R",
    "RBI": "RBI",
    "BB": "BB",
    "SO": "SO",
    "HBP": "HBP",
    "SB": "SB",
    "CS": "CS",
    "AVG": "AVG",
    "OBP": "OBP",
    "SLG": "SLG",
    "OPS": "OPS",
    "wOBA": "wOBA",
    "wRC+": "wRC+",
    "WAR": "WAR",
    "playerid": "fg_id",
    "PlayerId": "fg_id",
}

PITCHER_COLUMN_MAP = {
    "Name": "name",
    "Team": "team",
    "W": "W",
    "L": "L",
    "G": "G",
    "GS": "GS",
    "IP": "IP",
    "H": "H_allowed",
    "ER": "ER",
    "HR": "HR_allowed",
    "BB": "BB",
    "SO": "K",
    "K": "K",
    "ERA": "ERA",
    "WHIP": "WHIP",
    "K/9": "K9",
    "BB/9": "BB9",
    "FIP": "FIP",
    "WAR": "WAR",
    "SV": "SV",
    "HLD": "HLD",
    "playerid": "fg_id",
    "PlayerId": "fg_id",
}

# ESPN position ID to name mapping
ESPN_POSITION_MAP = {
    0: "C", 1: "1B", 2: "2B", 3: "3B", 4: "SS",
    5: "OF", 6: "OF", 7: "OF",  # LF, CF, RF all map to OF
    8: "UTIL", 9: "SP", 10: "RP", 11: "RP",
    12: "DH", 13: "UTIL", 14: "SP",
}


def load_hitter_projections(
    file_content: str | bytes,
    system: ProjectionSystem = ProjectionSystem.CUSTOM,
    position_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Load hitter projections from CSV content.

    Args:
        file_content: CSV file content (string or bytes).
        system: Which projection system this data comes from.
        position_data: Optional DataFrame with player positions.

    Returns:
        Normalized DataFrame with required columns for SGP calculation.
    """
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")

    df = pd.read_csv(io.StringIO(file_content))

    # Normalize column names
    rename_map = {k: v for k, v in HITTER_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["name", "AB", "H", "HR", "R", "RBI", "SB"]
    for col in required:
        if col not in df.columns:
            if col == "name" and "Name" in df.columns:
                df["name"] = df["Name"]
            else:
                df[col] = 0

    # Compute derived fields
    if "AVG" not in df.columns and "H" in df.columns and "AB" in df.columns:
        df["AVG"] = df["H"] / df["AB"].replace(0, 1)

    if "PA" not in df.columns:
        df["PA"] = df.get("AB", 0) + df.get("BB", 0) + df.get("HBP", 0)

    # Set player type and projection system
    df["player_type"] = PlayerType.HITTER.value
    df["projection_system"] = system.value

    # Handle positions
    if position_data is not None and "position" not in df.columns:
        df = _merge_positions(df, position_data)
    elif "position" not in df.columns:
        # Try to infer from Pos column
        if "Pos" in df.columns:
            df["position"] = df["Pos"].apply(_normalize_position)
        else:
            df["position"] = "UTIL"

    # Set eligible positions as list
    if "eligible_positions" not in df.columns:
        df["eligible_positions"] = df["position"].apply(lambda p: [p] if isinstance(p, str) else p)

    # Create unique player ID if not present
    if "player_id" not in df.columns:
        if "fg_id" in df.columns:
            df["player_id"] = df["fg_id"].astype(str)
        else:
            df["player_id"] = df["name"].str.lower().str.replace(r"\s+", "_", regex=True)

    df = df.set_index("player_id", drop=False)

    return df


def load_pitcher_projections(
    file_content: str | bytes,
    system: ProjectionSystem = ProjectionSystem.CUSTOM,
) -> pd.DataFrame:
    """
    Load pitcher projections from CSV content.

    Args:
        file_content: CSV file content (string or bytes).
        system: Which projection system this data comes from.

    Returns:
        Normalized DataFrame with required columns for SGP calculation.
    """
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")

    df = pd.read_csv(io.StringIO(file_content))

    # Normalize column names
    rename_map = {k: v for k, v in PITCHER_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Handle SO -> K rename (FanGraphs uses both)
    if "K" not in df.columns and "SO" in df.columns:
        df["K"] = df["SO"]

    # Ensure required columns exist
    required = ["name", "IP", "W", "K", "ERA", "WHIP"]
    for col in required:
        if col not in df.columns:
            if col == "name" and "Name" in df.columns:
                df["name"] = df["Name"]
            else:
                df[col] = 0

    # Compute derived fields
    if "SV" not in df.columns:
        df["SV"] = 0

    if "ER" not in df.columns and "ERA" in df.columns:
        df["ER"] = df["ERA"] * df["IP"] / 9

    if "BB_plus_H" not in df.columns:
        df["BB_plus_H"] = df["WHIP"] * df["IP"]

    if "GS" not in df.columns:
        df["GS"] = 0

    # Determine SP vs RP
    if "position" not in df.columns:
        df["position"] = df.apply(
            lambda row: "SP" if row.get("GS", 0) > 5 else "RP", axis=1
        )

    df["eligible_positions"] = df["position"].apply(lambda p: [p])
    df["player_type"] = PlayerType.PITCHER.value
    df["projection_system"] = system.value
    df["starts"] = df.get("GS", 0)

    # Create unique player ID
    if "player_id" not in df.columns:
        if "fg_id" in df.columns:
            df["player_id"] = df["fg_id"].astype(str)
        else:
            df["player_id"] = df["name"].str.lower().str.replace(r"\s+", "_", regex=True)

    df = df.set_index("player_id", drop=False)

    return df


def combine_projections(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine hitter and pitcher projections into a single player pool.

    Fills missing category columns with 0 so SGP calculations work
    across all players.
    """
    # Ensure both DataFrames have all category columns
    all_cats = ["R", "HR", "RBI", "SB", "AVG", "W", "SV", "K", "ERA", "WHIP"]
    for cat in all_cats:
        if cat not in hitters.columns:
            hitters[cat] = 0
        if cat not in pitchers.columns:
            pitchers[cat] = 0

    # For rate stats, set sensible defaults for the other player type
    # Hitters don't have ERA/WHIP; pitchers don't have AVG
    hitters.loc[:, "ERA"] = 0
    hitters.loc[:, "WHIP"] = 0
    hitters.loc[:, "IP"] = 0
    pitchers.loc[:, "AVG"] = 0
    pitchers.loc[:, "AB"] = 0

    combined = pd.concat([hitters, pitchers], ignore_index=False)

    # Handle duplicate indices
    if combined.index.duplicated().any():
        combined = combined[~combined.index.duplicated(keep="first")]

    return combined


def add_adp_data(
    player_pool: pd.DataFrame,
    adp_data: pd.DataFrame | dict,
) -> pd.DataFrame:
    """
    Merge ADP (Average Draft Position) data into the player pool.

    ADP data should have player_id or name as key, with columns:
        - adp: Average draft position
        - adp_std: Standard deviation of draft position (optional)
        - adp_rank: Rank by ADP (computed if not provided)
    """
    df = player_pool.copy()

    if isinstance(adp_data, dict):
        adp_df = pd.DataFrame(adp_data)
    else:
        adp_df = adp_data.copy()

    # Try to merge on player_id first, then name
    if "player_id" in adp_df.columns:
        adp_df = adp_df.set_index("player_id")
        for col in ["adp", "adp_std"]:
            if col in adp_df.columns:
                df[col] = adp_df[col]
    elif "name" in adp_df.columns:
        adp_lookup = adp_df.set_index("name")
        name_map = df["name"].to_dict()
        for pid, name in name_map.items():
            if name in adp_lookup.index:
                for col in ["adp", "adp_std"]:
                    if col in adp_lookup.columns:
                        df.loc[pid, col] = adp_lookup.loc[name, col]

    # Fill missing ADP with large value (undrafted)
    if "adp" not in df.columns:
        df["adp"] = 999
    df["adp"] = df["adp"].fillna(999)

    if "adp_std" not in df.columns:
        # Estimate ADP standard deviation (~15% of ADP, minimum 3)
        df["adp_std"] = np.maximum(df["adp"] * 0.15, 3.0)

    # Compute ADP rank
    df["adp_rank"] = df["adp"].rank(method="min")

    return df


def _normalize_position(pos_str: str) -> str:
    """Normalize position string to standard format."""
    pos = pos_str.strip().upper()
    position_aliases = {
        "LF": "OF", "CF": "OF", "RF": "OF",
        "DH": "UTIL",
        "MI": "SS", "CI": "1B",
    }
    return position_aliases.get(pos, pos)


def _merge_positions(df: pd.DataFrame, position_data: pd.DataFrame) -> pd.DataFrame:
    """Merge position data into player projections."""
    if "name" in position_data.columns:
        pos_map = position_data.set_index("name")["position"].to_dict()
        df["position"] = df["name"].map(pos_map).fillna("UTIL")
    return df


# NumPy import at module level (used in add_adp_data)
import numpy as np
