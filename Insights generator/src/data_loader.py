from __future__ import annotations

from typing import Optional, List, Dict
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Optional: pull schema config from config.py if you add it there later
# ---------------------------------------------------------------------
try:
    # If you define these in config.py, this loader will use them.
    from .config import CANONICAL_ALIASES, MIN_REQUIRED, NUMERIC_COLUMNS
except Exception:
    # Fallback defaults so this file works immediately.
    CANONICAL_ALIASES = {
        # Dimensions
        "Campaign": ["Campaign", "Belongs_to_this_Campaign"],
        "Platform": ["Platform"],
        "Market": ["Market", "Market_Area", "Geo", "Region", "Country"],
        "Channel": ["Channel", "Marketing_Channel_Type", "Marketing Channel Type"],
        "Device": ["Device"],

        # Time (do NOT map Week_Start here – we handle it separately)
        "Date": ["Date"],
        "Week": ["Week", "ISO_Week"],
        "Quarter": ["Quarter"],
        "Year": ["Year"],

        # Core metrics
        "Spend": ["Spend", "Actual_Weekly_Spend", "Spend_Weekly_Actual", "Spend_Daily_Actual"],
        "Traffic": ["Traffic", "Traffic_Weekly_Actual", "Traffic_Daily_Actual", "Visits"],
        "Clicks": ["Clicks", "Clicks_Weekly_Actual", "Clicks_Daily_Actual"],
        "Impressions": ["Impressions", "Impressions_Weekly_Actual", "Impressions_Daily_Actual"],
        "Engaged Visits": ["Engaged Visits", "~Engaged Visits", "Engaged_Visits"],

        # Optional social/video-like metrics
        "Video Start": ["Video Start", "Video Starts"],
        "Video End": ["Video End", "Video Ends"],
        "Video Completion Rate": ["Video Completion Rate"],
        "Likes/Reactions": ["Likes/Reactions", "Likes Reactions", "Reactions", "Likes"],
        "Shares": ["Shares"],
        "Comments": ["Comments"],
        "Non-GTM Total Seconds Spent": ["Non-GTM Total Seconds Spent"],
    }

    MIN_REQUIRED = [
        "Campaign",
        "Platform",
        "Spend",
        "Traffic",
    ]

    # Keep this aligned with whatever you already used in your project.
    NUMERIC_COLUMNS = [
        "Spend",
        "Traffic",
        "Clicks",
        "Impressions",
        "Engaged Visits",
        "Video Start",
        "Video End",
        "Video Completion Rate",
        "Likes/Reactions",
        "Shares",
        "Comments",
        "Non-GTM Total Seconds Spent",
    ]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Strip whitespace and unify accidental double spaces
    df.columns = [str(c).strip().replace("  ", " ") for c in df.columns]
    return df


def _derive_adobe_fiscal_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Year / Week / Quarter using Adobe fiscal fields when present.
    Example:
        Adobe_FY_Week = "2026-01"  -> Year = 2026, Week = 1
        Quarter       = "FY26Q1"  -> Quarter = 1, Year_FY = 2026
    """
    df = df.copy()

    # Use Week_Start as a canonical Date reference if present and Date not set
    if "Week_Start" in df.columns and "Date" not in df.columns:
        df["Date"] = df["Week_Start"]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # 1) Adobe_FY_Week like "2026-01"
    if "Adobe_FY_Week" in df.columns:
        tmp = df["Adobe_FY_Week"].astype(str).str.extract(
            r"(?P<year>\d{4})[-_/]?(?P<week>\d{1,2})"
        )
        fy_year = pd.to_numeric(tmp["year"], errors="coerce").astype("Int64")
        fy_week = pd.to_numeric(tmp["week"], errors="coerce").astype("Int64")

        # Overwrite canonical Year/Week with fiscal versions
        df["Year"] = fy_year
        df["Week"] = fy_week

    # 2) Quarter like "FY26Q1"
    if "Quarter" in df.columns:
        qtmp = df["Quarter"].astype(str).str.extract(
            r"FY(?P<fy>\d{2})Q(?P<q>\d)"
        )
        q_num = pd.to_numeric(qtmp["q"], errors="coerce").astype("Int64")
        fy2 = pd.to_numeric(qtmp["fy"], errors="coerce").astype("Int64")
        fy_full = (2000 + fy2).astype("Int64")

        # Replace Quarter string with numeric quarter where we can parse it
        # (leave as-is where parsing fails)
        df["Quarter"] = q_num.where(~q_num.isna(), df["Quarter"])

        # If Year wasn't set from Adobe_FY_Week, set it from FYxx
        if "Year" not in df.columns or df["Year"].isna().all():
            df["Year"] = fy_full

        # Optional helper for debugging / reference
        df["Year_FY"] = fy_full

    return df


def _derive_time_from_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback: derive calendar ISO Year/Week/Quarter from Date
    ONLY where they don't already exist (so we don't overwrite fiscal).
    """
    df = df.copy()
    if "Date" not in df.columns:
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Only derive missing pieces
    if "Year" not in df.columns or "Week" not in df.columns:
        iso = df["Date"].dt.isocalendar()
        if "Year" not in df.columns:
            df["Year"] = iso.year.astype("Int64")
        if "Week" not in df.columns:
            df["Week"] = iso.week.astype("Int64")

    if "Quarter" not in df.columns:
        df["Quarter"] = df["Date"].dt.quarter.astype("Int64")

    return df


def _parse_yearweek_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a dataset provides a YearWeek string but not Year/Week,
    attempt to parse formats like:
      - '2025-W1'
      - '2025-1'
      - '2025_W01'
    """
    df = df.copy()
    if "YearWeek" not in df.columns:
        return df

    if "Year" in df.columns and "Week" in df.columns:
        return df

    s = df["YearWeek"].astype(str)

    # Try to extract year/week numbers
    extracted = s.str.extract(r"(?P<year>\d{4}).*?(?P<week>\d{1,2})")
    if "Year" not in df.columns:
        df["Year"] = pd.to_numeric(extracted["year"], errors="coerce").astype("Int64")
    if "Week" not in df.columns:
        df["Week"] = pd.to_numeric(extracted["week"], errors="coerce").astype("Int64")

    return df


# ---------------------------------------------------------------------
# Schema Normalisation
# ---------------------------------------------------------------------

def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert different input datasets into a canonical schema expected
    by the rest of your pipeline (features, filters, insights).

    Priority:
      1) Rename columns to canonical names.
      2) Derive Adobe fiscal Year/Week/Quarter if relevant fields exist.
      3) Fallback to calendar ISO Year/Week/Quarter from Date.
      4) Fallback to YearWeek string if needed.
    """
    df = _normalise_column_names(df)

    # 1) Rename alias → canonical
    rename_map: Dict[str, str] = {}
    for canonical, alts in CANONICAL_ALIASES.items():
        found = _pick_first_existing(df, alts)
        if found and found != canonical:
            rename_map[found] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)

    # 2) Adobe fiscal time (if Adobe_FY_Week / FYxxQy present)
    df = _derive_adobe_fiscal_time(df)

    # 3) Fallback: derive from Date if still missing
    df = _derive_time_from_date(df)

    # 4) Fallback: parse YearWeek if provided
    df = _parse_yearweek_string(df)

    # 5) Add safe default dimension cols to prevent groupby failures
    for dim in ["Channel", "Market", "Device"]:
        if dim not in df.columns:
            df[dim] = "Unknown"

    # 6) Coerce time columns
    for col in ["Year", "Quarter", "Week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # 7) Coerce numerics you care about
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = _safe_to_num(df[col])

    # 8) Validate minimal required canonical columns
    missing = [c for c in MIN_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing minimum required columns after normalisation: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # 9) Generate YearWeek helper
    if "Year" in df.columns and "Week" in df.columns:
        df["YearWeek"] = df["Year"].astype(str) + "-W" + df["Week"].astype(str)
    elif "Date" in df.columns:
        # fallback YearWeek from date if needed
        iso = df["Date"].dt.isocalendar()
        df["YearWeek"] = iso.year.astype(str) + "-W" + iso.week.astype(str)
    else:
        df["YearWeek"] = None

    return df


# ---------------------------------------------------------------------
# Public Loader
# ---------------------------------------------------------------------

def load_data(
    path: str | Path,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load CSV/XLSX and normalise to canonical schema.

    Args:
        path: file path
        sheet_name: optional Excel sheet name

    Returns:
        Normalised DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif suffix in [".xlsx", ".xls"]:
        # default: first sheet if none specified
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # If sheet_name=None, pandas returns a DataFrame.
    # If sheet_name is a list/dict logic is used elsewhere;
    # we only support a single-sheet load here.
    if isinstance(df, dict):
        # Pick the first sheet
        first_key = next(iter(df.keys()))
        df = df[first_key]

    df = normalise_schema(df)
    return df
