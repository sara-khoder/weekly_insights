from __future__ import annotations

from typing import Optional
import pandas as pd

from .config import FilterConfig


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _safe_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def _ensure_time_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure Year/Quarter/Week are numeric where present.
    Doesn't create them (that's the loader's job).
    """
    df = df.copy()
    for c in ["Year", "Quarter", "Week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _apply_list_filter(df: pd.DataFrame, col: str, values) -> pd.DataFrame:
    """
    Apply an IN filter only if col exists and values is non-empty.
    """
    if not values:
        return df
    if col not in df.columns:
        return df
    return df[df[col].isin(values)]


# ---------------------------------------------------------------------
# Core filter logic
# ---------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, cfg: Optional[FilterConfig] = None) -> pd.DataFrame:
    """
    Applies FilterConfig to a dataframe.

    Defensive behaviour:
    - If a filter column doesn't exist, it is ignored.
    - If time columns contain NaNs, filters operate on available rows.
    - If cfg is None, returns df unchanged.
    """
    if cfg is None:
        return df

    out = df.copy()
    out = _ensure_time_types(out)

    # -------------------------
    # Time filters
    # -------------------------

    # Years filter
    if cfg.years and _has_col(out, "Year"):
        out = out[out["Year"].isin(cfg.years)]

    # Quarter filter
    if cfg.quarter is not None and _has_col(out, "Quarter"):
        q = _safe_int(cfg.quarter)
        if q is not None:
            out = out[out["Quarter"] == q]

    # Week range filters
    # These make the most sense within a single year/quarter context,
    # but we apply them safely regardless.
    if cfg.start_week is not None and _has_col(out, "Week"):
        sw = _safe_int(cfg.start_week)
        if sw is not None:
            out = out[out["Week"] >= sw]

    if cfg.end_week is not None and _has_col(out, "Week"):
        ew = _safe_int(cfg.end_week)
        if ew is not None:
            out = out[out["Week"] <= ew]

    # -------------------------
    # Dimension filters
    # -------------------------
    out = _apply_list_filter(out, "Campaign", cfg.campaigns)
    out = _apply_list_filter(out, "Market", cfg.markets)
    out = _apply_list_filter(out, "Channel", cfg.channels)
    out = _apply_list_filter(out, "Device", cfg.devices)
    out = _apply_list_filter(out, "Platform", cfg.platforms)

    return out


# ---------------------------------------------------------------------
# Convenience wrappers (aliases)
# These help avoid breaking older imports in your project.
# ---------------------------------------------------------------------

def filter_dataframe(df: pd.DataFrame, cfg: Optional[FilterConfig] = None) -> pd.DataFrame:
    """Alias for apply_filters."""
    return apply_filters(df, cfg)


def apply_filter_config(df: pd.DataFrame, cfg: Optional[FilterConfig] = None) -> pd.DataFrame:
    """Alias for apply_filters."""
    return apply_filters(df, cfg)


# ---------------------------------------------------------------------
# Optional: split helpers if your pipeline benefits from them
# ---------------------------------------------------------------------

def filter_raw(df: pd.DataFrame, cfg: Optional[FilterConfig] = None) -> pd.DataFrame:
    """
    For raw (pre-aggregation) datasets.
    Currently same logic as apply_filters, but kept for clarity/future tweaks.
    """
    return apply_filters(df, cfg)


def filter_weekly(df: pd.DataFrame, cfg: Optional[FilterConfig] = None) -> pd.DataFrame:
    """
    For weekly aggregated datasets.
    Same logic, but kept as a semantic wrapper.
    """
    return apply_filters(df, cfg)
