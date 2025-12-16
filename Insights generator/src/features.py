from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _to_num(s):
    """Safely coerce a Series to numeric."""
    return pd.to_numeric(s, errors="coerce")


def _ensure_cols(df: pd.DataFrame, cols: List[str], fill=np.nan) -> pd.DataFrame:
    """Ensure columns exist in df; create with fill if missing."""
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df


# ---------------------------------------------------------------------
# Derived metrics at row-level (pre-aggregation)
# ---------------------------------------------------------------------

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calculated KPI columns if the required raw columns exist.
    These are row-level calculations; weekly/QTD calculations are
    computed again after aggregation.
    """
    df = df.copy()

    # CTR
    if "Clicks" in df.columns and "Impressions" in df.columns:
        clicks = _to_num(df["Clicks"])
        imps = _to_num(df["Impressions"])
        df["CTR_calc"] = np.where(imps > 0, clicks / imps, np.nan)

    # Engaged Visit Rate (EVR)
    if "Engaged Visits" in df.columns and "Traffic" in df.columns:
        ev = _to_num(df["Engaged Visits"])
        traffic = _to_num(df["Traffic"])
        df["EVR_calc"] = np.where(traffic > 0, ev / traffic, np.nan)

    # Cost Per Engaged Visit (CPE)
    if "Spend" in df.columns and "Engaged Visits" in df.columns:
        spend = _to_num(df["Spend"])
        ev = _to_num(df["Engaged Visits"])
        df["CPE_calc"] = np.where(ev > 0, spend / ev, np.nan)

    return df


# ---------------------------------------------------------------------
# Aggregation config
# ---------------------------------------------------------------------
# Output column -> (raw column, agg function)
RAW_TO_AGG: Dict[str, Tuple[str, str]] = {
    "Spend": ("Spend", "sum"),
    "Traffic": ("Traffic", "sum"),
    "Engaged_Visits": ("Engaged Visits", "sum"),
    "Impressions": ("Impressions", "sum"),
    "Clicks": ("Clicks", "sum"),
}


# ---------------------------------------------------------------------
# Weekly Aggregation
# ---------------------------------------------------------------------

def aggregate_weekly(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Aggregates data to weekly granularity, optionally split by group_cols.
    Produces safe derived metrics and WoW deltas.
    """
    df = df.copy()

    # Basic time guard
    missing_time = [c for c in ["Year", "Quarter", "Week"] if c not in df.columns]
    if missing_time:
        raise ValueError(
            f"aggregate_weekly requires time columns {missing_time}. "
            "Make sure your loader/normaliser derives these from Date if needed."
        )

    group_keys = ["Year", "Quarter", "Week"] + group_cols

    # Build dynamic agg spec based on availability
    agg_spec = {}
    for out_col, (raw_col, fn) in RAW_TO_AGG.items():
        if raw_col in df.columns:
            agg_spec[out_col] = (raw_col, fn)

    # If no metric columns exist, still return unique group rows
    if not agg_spec:
        agg = df[group_keys].drop_duplicates().copy()
        agg = _ensure_cols(agg, list(RAW_TO_AGG.keys()))
    else:
        agg = df.groupby(group_keys, dropna=False).agg(**agg_spec).reset_index()
        agg = _ensure_cols(agg, list(RAW_TO_AGG.keys()))

    # Coerce numerics for safe maths
    for c in ["Spend", "Traffic", "Engaged_Visits", "Impressions", "Clicks"]:
        agg[c] = _to_num(agg[c])

    # Derived weekly KPIs
    agg["EVR"] = np.where(agg["Traffic"] > 0, agg["Engaged_Visits"] / agg["Traffic"], np.nan)
    agg["CTR"] = np.where(agg["Impressions"] > 0, agg["Clicks"] / agg["Impressions"], np.nan)
    agg["CPE"] = np.where(agg["Engaged_Visits"] > 0, agg["Spend"] / agg["Engaged_Visits"], np.nan)

    # Sorting + IDs
    agg = agg.sort_values(["Year", "Quarter", "Week"] + group_cols).reset_index(drop=True)
    agg["Week_ID"] = agg["Year"].astype(str).str.cat(agg["Week"].astype(str), sep="-")

    # WoW deltas
    wow_cols = ["Spend", "Traffic", "Engaged_Visits", "Impressions", "Clicks", "EVR", "CTR", "CPE"]

    if group_cols:
        for c in wow_cols:
            agg[f"{c}_WoW"] = agg.groupby(group_cols, dropna=False)[c].pct_change()
    else:
        for c in wow_cols:
            agg[f"{c}_WoW"] = agg[c].pct_change()

    return agg


# ---------------------------------------------------------------------
# QTD Aggregation (from weekly)
# ---------------------------------------------------------------------

def aggregate_qtd(weekly_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Builds Quarter-to-Date aggregates from weekly aggregated df.
    """
    weekly_df = weekly_df.copy()

    group_keys = ["Year", "Quarter"] + group_cols

    # Ensure base metric cols exist
    weekly_df = _ensure_cols(weekly_df, list(RAW_TO_AGG.keys()))

    for c in ["Spend", "Traffic", "Engaged_Visits", "Impressions", "Clicks"]:
        weekly_df[c] = _to_num(weekly_df[c])

    qtd = (
        weekly_df.groupby(group_keys, dropna=False)
        .agg(
            Spend_QTD=("Spend", "sum"),
            Traffic_QTD=("Traffic", "sum"),
            Engaged_Visits_QTD=("Engaged_Visits", "sum"),
            Impressions_QTD=("Impressions", "sum"),
            Clicks_QTD=("Clicks", "sum"),
        )
        .reset_index()
    )

    # Derived QTD KPIs
    qtd["EVR_QTD"] = np.where(qtd["Traffic_QTD"] > 0, qtd["Engaged_Visits_QTD"] / qtd["Traffic_QTD"], np.nan)
    qtd["CTR_QTD"] = np.where(qtd["Impressions_QTD"] > 0, qtd["Clicks_QTD"] / qtd["Impressions_QTD"], np.nan)
    qtd["CPE_QTD"] = np.where(qtd["Engaged_Visits_QTD"] > 0, qtd["Spend_QTD"] / qtd["Engaged_Visits_QTD"], np.nan)

    return qtd


# ---------------------------------------------------------------------
# Public API used by the pipeline
# ---------------------------------------------------------------------

def build_all_aggregates(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Builds the full set of aggregates expected by the pipeline.
    This function is defensive: it ensures grouping dimensions exist
    to avoid crashing when running on a slightly different dataset.
    """
    df = df.copy()

    # Ensure common dimensions exist so groupby doesn't break
    for dim in ["Channel", "Market", "Device"]:
        if dim not in df.columns:
            df[dim] = "Unknown"

    df = add_derived_metrics(df)

    # Weekly
    overall_weekly = aggregate_weekly(df, group_cols=[])
    channel_weekly = aggregate_weekly(df, group_cols=["Channel"])
    market_weekly = aggregate_weekly(df, group_cols=["Market"])
    device_weekly = aggregate_weekly(df, group_cols=["Device"])

    # QTD
    overall_qtd = aggregate_qtd(overall_weekly, group_cols=[])
    channel_qtd = aggregate_qtd(channel_weekly, group_cols=["Channel"])
    market_qtd = aggregate_qtd(market_weekly, group_cols=["Market"])
    device_qtd = aggregate_qtd(device_weekly, group_cols=["Device"])

    return {
        "overall_weekly": overall_weekly,
        "channel_weekly": channel_weekly,
        "market_weekly": market_weekly,
        "device_weekly": device_weekly,
        "overall_qtd": overall_qtd,
        "channel_qtd": channel_qtd,
        "market_qtd": market_qtd,
        "device_qtd": device_qtd,
    }
