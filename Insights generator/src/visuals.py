from __future__ import annotations

from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# Colour system (vibrant, stakeholder-friendly)
# -----------------------------------------------------------------------------

VIBRANT_SEQ = [
    "#22C55E",  # green
    "#3B82F6",  # blue
    "#F97316",  # orange
    "#A855F7",  # purple
    "#06B6D4",  # cyan
    "#F43F5E",  # pink/red
    "#EAB308",  # yellow
    "#6366F1",  # indigo
    "#14B8A6",  # teal
    "#84CC16",  # lime
]

def _cycle_colors(n: int, palette=None):
    palette = palette or VIBRANT_SEQ
    if n <= 0:
        return []
    return [palette[i % len(palette)] for i in range(n)]



# ---------------------------------------------------------------------
# Base Plotly styling
# ---------------------------------------------------------------------

def _base_layout(fig: go.Figure, y_title: str):
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title=y_title,
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_traces(marker=dict(size=7), line=dict(width=3))
    return fig


import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

PLOT_FONT = dict(family="Inter, system-ui, sans-serif", size=12)

def apply_modern_layout(fig: go.Figure, title: str = "", x_title: str = "", y_title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        font=PLOT_FONT,
        margin=dict(l=40, r=20, t=60, b=40),
        colorway=VIBRANT_SEQ,  # ✅ global palette
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig

# -----------------------------------------------------------------------------
# NEW: Latest-week comparison visuals (story-first)
# -----------------------------------------------------------------------------

def _get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _latest_week_slice(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Returns df filtered to the latest (Year, Quarter, Week) combination present
    in df_raw, plus a human label like 'Q1-W1'.

    Robust to single-week selections.
    """
    if df_raw is None or df_raw.empty:
        return df_raw, "Latest week"

    needed = {"Year", "Quarter", "Week"}
    if not needed.issubset(set(df_raw.columns)):
        return df_raw, "Latest week"

    d = df_raw.dropna(subset=["Year", "Quarter", "Week"]).copy()
    if d.empty:
        return df_raw, "Latest week"

    d = d.sort_values(["Year", "Quarter", "Week"])
    latest = d.tail(1).iloc[0]
    y = int(latest["Year"])
    q = int(latest["Quarter"])
    w = int(latest["Week"])

    label = f"Q{q}-W{w}"
    latest_df = d[(d["Year"] == y) & (d["Quarter"] == q) & (d["Week"] == w)].copy()
    return latest_df, label


def latest_week_dimension_bar(
    df_raw: pd.DataFrame,
    dimension: str,
    metric_key: str,
    title: Optional[str] = None,
    top_n: int = 10,
) -> go.Figure:
    """
    Bar chart comparing a metric by dimension for the latest week in df_raw.

    metric_key supported:
      - "Spend"
      - "Traffic"
      - "Engaged Visits" / "Engaged_Visits"
      - "EVR"
      - "CPV"
      - "CPE"
    """
    if df_raw is None or df_raw.empty or dimension not in df_raw.columns:
        return go.Figure()

    spend_col = _get_col(df_raw, ["Spend"])
    traffic_col = _get_col(df_raw, ["Traffic"])
    engaged_col = _get_col(df_raw, ["Engaged Visits", "Engaged_Visits"])

    df_latest, week_label = _latest_week_slice(df_raw)
    if df_latest is None or df_latest.empty:
        return go.Figure()

    # Aggregate base measures by dimension
    agg_dict = {}
    if spend_col:
        agg_dict["Spend"] = (spend_col, "sum")
    if traffic_col:
        agg_dict["Traffic"] = (traffic_col, "sum")
    if engaged_col:
        agg_dict["Engaged"] = (engaged_col, "sum")

    if not agg_dict:
        return go.Figure()

    d = (
        df_latest.groupby(dimension, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    # Derived metrics
    if "Traffic" in d.columns and "Engaged" in d.columns:
        d["EVR"] = np.where(d["Traffic"] > 0, d["Engaged"] / d["Traffic"], np.nan)
        d["CPV"] = np.where((d.get("Spend", 0) > 0) & (d["Traffic"] > 0), d["Spend"] / d["Traffic"], np.nan)
        d["CPE"] = np.where((d.get("Spend", 0) > 0) & (d["Engaged"] > 0), d["Spend"] / d["Engaged"], np.nan)

    # Map metric_key to column
    key_map = {
        "Spend": "Spend",
        "Traffic": "Traffic",
        "Engaged Visits": "Engaged",
        "Engaged_Visits": "Engaged",
        "EVR": "EVR",
        "CPV": "CPV",
        "CPE": "CPE",
    }
    metric_col = key_map.get(metric_key, metric_key)
    if metric_col not in d.columns:
        return go.Figure()

    # Sort + top_n
    d = d.sort_values(metric_col, ascending=False)
    if top_n and len(d) > top_n:
        d = d.head(top_n)

    plot_title = title or f"{metric_key} by {dimension} ({week_label})"

    # Create bar chart WITHOUT text_auto – we’ll control labels ourselves
    fig = px.bar(
        d,
        x=dimension,
        y=metric_col,
        title=plot_title,
    )
    fig.update_traces(marker_line_width=0.3)

    # Multi-colour bars (one colour per bar)
    bar_colors = _cycle_colors(len(d))
    fig.update_traces(
        marker_color=bar_colors,
        marker_line_width=0.6,
        marker_line_color="rgba(0,0,0,0.08)",
    )

    # ----- Label formatting -----
    if metric_col == "EVR":
        # EVR is a rate (0–1) → show as %
        fig.update_traces(
            texttemplate="%{y:.1%}",
            textposition="outside",
            hovertemplate=f"%{{x}}<br>{metric_key}: %{{y:.2%}}<extra></extra>",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        y_title = "EVR (%)"
    else:
        # All other metrics as whole numbers with thousands separator
        fig.update_traces(
            texttemplate="%{y:,.0f}",
            textposition="outside",
            hovertemplate=f"%{{x}}<br>{metric_key}: %{{y:,.0f}}<extra></extra>",
        )
        y_title = metric_key

    fig = apply_modern_layout(fig, title=plot_title, x_title=dimension, y_title=y_title)
    return fig


def latest_week_dimension_pie(
    df_raw: pd.DataFrame,
    dimension: str,
    metric_key: str,
    title: Optional[str] = None,
    top_n: int = 6,
) -> go.Figure:
    """
    Pie chart share view for latest week.
    Best for Spend / Traffic / Engaged Visits.
    """
    if df_raw is None or df_raw.empty or dimension not in df_raw.columns:
        return go.Figure()

    df_latest, week_label = _latest_week_slice(df_raw)
    if df_latest is None or df_latest.empty:
        return go.Figure()

    spend_col = _get_col(df_raw, ["Spend"])
    traffic_col = _get_col(df_raw, ["Traffic"])
    engaged_col = _get_col(df_raw, ["Engaged Visits", "Engaged_Visits"])

    base_map = {
        "Spend": spend_col,
        "Traffic": traffic_col,
        "Engaged Visits": engaged_col,
        "Engaged_Visits": engaged_col,
    }
    src = base_map.get(metric_key)
    if not src:
        return go.Figure()

    d = (
        df_latest.groupby(dimension, dropna=False)
        .agg(Value=(src, "sum"))
        .reset_index()
        .sort_values("Value", ascending=False)
    )

    if top_n and len(d) > top_n:
        top = d.head(top_n).copy()
        other = pd.DataFrame({dimension: ["Other"], "Value": [d["Value"].iloc[top_n:].sum()]})
        d = pd.concat([top, other], ignore_index=True)

    plot_title = title or f"{metric_key} share by {dimension} ({week_label})"

    fig = px.pie(
        d,
        names=dimension,
        values="Value",
        title=plot_title,
        hole=0.35,
        color_discrete_sequence=VIBRANT_SEQ,
    )
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig



def weekly_kpi_strip(overall_weekly: pd.DataFrame) -> go.Figure:
    """
    Builds a slick KPI strip comparing latest week vs previous week.
    Expects overall_weekly to have:
      Spend, Traffic, Engaged_Visits, EVR, CPV, CPE
    """
    if overall_weekly is None or overall_weekly.empty:
        return go.Figure()

    d = overall_weekly.sort_values(["Year", "Quarter", "Week"]).copy()
    latest = d.tail(1)
    prev = d.tail(2).head(1) if len(d) >= 2 else None

    def _get(col):
        v = latest[col].iloc[0] if col in latest.columns else np.nan
        p = prev[col].iloc[0] if (prev is not None and col in prev.columns) else np.nan
        return v, p

    metrics = [
        ("Spend", "£"),
        ("Traffic", ""),
        ("Engaged_Visits", ""),
        ("EVR", ""),
        ("CPV", "£"),
        ("CPE", "£"),
    ]

    fig = go.Figure()

    # Create six indicator boxes
    for i, (m, prefix) in enumerate(metrics):
        v, p = _get(m)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=float(v) if pd.notna(v) else 0,
                number={"prefix": prefix} if prefix else {},
                delta={"reference": float(p)} if pd.notna(p) else {},
                title={"text": m.replace("_", " ")},
                domain={"row": 0, "column": i},
            )
        )

    fig.update_layout(
        grid={"rows": 1, "columns": len(metrics), "pattern": "independent"},
        template="plotly_white",
        font=PLOT_FONT,
        margin=dict(l=20, r=20, t=40, b=10),
        height=140,
    )

    return fig

def anomaly_timeline(overall_weekly: pd.DataFrame, title: str = "Spend with flagged anomalies") -> go.Figure:
    if overall_weekly is None or overall_weekly.empty:
        return go.Figure()

    d = overall_weekly.sort_values(["Year", "Quarter", "Week"]).copy()
    d["Week_Label"] = "Q" + d["Quarter"].astype(str) + "-W" + d["Week"].astype(str)

    fig = px.line(
        d,
        x="Week_Label",
        y="Spend",
        markers=True,
        title=title,
    )

    if "is_anomaly" in d.columns:
        anomalies = d[d["is_anomaly"] == True]
        if not anomalies.empty:
            fig.add_scatter(
                x=anomalies["Week_Label"],
                y=anomalies["Spend"],
                mode="markers",
                name="Flagged",
                marker=dict(size=12, symbol="diamond"),
            )

    fig = apply_modern_layout(fig, title=title, x_title="", y_title="Spend")
    fig.update_xaxes(tickangle=0)
    return fig


def channel_efficiency_quadrant(channel_qtd: pd.DataFrame, title: str = "Channel efficiency (share view)") -> go.Figure:
    if channel_qtd is None or channel_qtd.empty:
        return go.Figure()

    d = channel_qtd.copy()

    total_spend = float(d["Spend_QTD"].sum()) if "Spend_QTD" in d.columns else 0.0
    total_eng = float(d["Engaged_Visits_QTD"].sum()) if "Engaged_Visits_QTD" in d.columns else 0.0
    total_spend = total_spend or 1.0
    total_eng = total_eng or 1.0

    d["Spend_share"] = d["Spend_QTD"] / total_spend
    d["Eng_share"] = d["Engaged_Visits_QTD"] / total_eng

    fig = px.scatter(
        d,
        x="Spend_share",
        y="Eng_share",
        size="Engaged_Visits_QTD",
        color="Channel",
        text="Channel",
        title=title,
    )

    fig.update_traces(textposition="top center")

    # 45-degree reference line
    max_val = float(max(d["Spend_share"].max(), d["Eng_share"].max())) * 1.1
    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=max_val, y1=max_val,
        line=dict(dash="dash"),
    )

    fig = apply_modern_layout(fig, title=title, x_title="Spend share", y_title="Engaged visits share")
    return fig


def campaign_micro_trends(df_raw: pd.DataFrame, metric: str = "Spend") -> go.Figure:
    """
    Small multiples by Campaign for quick visual scanning.
    """
    if df_raw is None or df_raw.empty:
        return go.Figure()

    needed = {"Campaign", "Year", "Quarter", "Week", metric}
    if not needed.issubset(set(df_raw.columns)):
        return go.Figure()

    d = (
        df_raw.groupby(["Year", "Quarter", "Week", "Campaign"], dropna=False)
        .agg(**{metric: (metric, "sum")})
        .reset_index()
        .sort_values(["Year", "Quarter", "Week"])
    )

    d["Week_Label"] = "Q" + d["Quarter"].astype(str) + "-W" + d["Week"].astype(str)

    fig = px.line(
        d,
        x="Week_Label",
        y=metric,
        facet_col="Campaign",
        facet_col_wrap=3,
        markers=True,
        title=f"{metric} micro-trends by campaign",
    )

    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        margin=dict(l=30, r=20, t=60, b=40),
        height=420,
        showlegend=False,
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(matches=None, showticklabels=False)
    return fig




# ---------------------------------------------------------------------
# Core charts used in app.py
# ---------------------------------------------------------------------

def line_metric_over_time(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    if df.empty or metric not in df.columns:
        return go.Figure()

    fig = px.line(
        df.sort_values(["Year", "Quarter", "Week"]),
        x="Week",
        y=metric,
        markers=True,
        title=title,
    )
    return _base_layout(fig, y_title=metric)


def bar_qtd_by_dimension(
    qtd_df: pd.DataFrame,
    dimension: str,
    value_col: str,
    title: str,
) -> go.Figure:
    if qtd_df.empty or dimension not in qtd_df.columns or value_col not in qtd_df.columns:
        return go.Figure()

    d = qtd_df.sort_values(value_col, ascending=False)
    fig = px.bar(
        d,
        x=dimension,
        y=value_col,
        title=title,
        text_auto=".2s",
    )
    fig.update_layout(
        xaxis_title=dimension,
        yaxis_title=value_col,
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_traces(marker_line_width=0.3)
    return fig


def stacked_share_bar(
    qtd_df: pd.DataFrame,
    dimension: str,
    value_cols: Dict[str, str],
    title: str,
) -> go.Figure:
    """Stacked bar showing share of different value_cols by dimension."""
    if qtd_df.empty or dimension not in qtd_df.columns:
        return go.Figure()

    d = qtd_df.copy()
    fig = go.Figure()
    for col, label in value_cols.items():
        if col not in d.columns:
            continue
        fig.add_bar(
            x=d[dimension],
            y=d[col],
            name=label,
        )

    fig.update_layout(
        barmode="relative",
        title=title,
        xaxis_title=dimension,
        yaxis_title="Share",
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def table_from_dataframe(df: pd.DataFrame, title: str, max_rows: int = 15) -> go.Figure:
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    d = df.head(max_rows).copy()

    header_values = list(d.columns)
    cell_values = [d[c].astype(str).tolist() for c in d.columns]

    # Alternating row fill
    n = len(d)
    zebra = ["#ffffff" if i % 2 == 0 else "#f9fafb" for i in range(n)]
    fill = [zebra for _ in d.columns]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_values,
                    align="left",
                    font=dict(family="Inter, system-ui, sans-serif", size=12, color="white"),
                    fill_color="#111827",
                ),
                cells=dict(
                    values=cell_values,
                    align="left",
                    font=dict(family="Inter, system-ui, sans-serif", size=11),
                    fill_color=fill,
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def efficiency_scatter(qtd_df: pd.DataFrame, dimension: str, title: str) -> go.Figure:
    """Scatter of spend share vs engagement share by dimension, sized by engaged visits."""
    if qtd_df.empty:
        return go.Figure()

    required = {"Spend_QTD", "Engaged_Visits_QTD", dimension}
    if not required.issubset(set(qtd_df.columns)):
        return go.Figure()

    d = qtd_df.copy()

    total_spend = float(d["Spend_QTD"].sum())
    total_eng = float(d["Engaged_Visits_QTD"].sum())
    if total_spend == 0:
        total_spend = 1.0
    if total_eng == 0:
        total_eng = 1.0

    d["Spend_share"] = d["Spend_QTD"] / total_spend
    d["Eng_share"] = d["Engaged_Visits_QTD"] / total_eng

    fig = px.scatter(
        d,
        x="Spend_share",
        y="Eng_share",
        size="Engaged_Visits_QTD",
        color=dimension,
        text=dimension,
        title=title,
    )
    fig.update_traces(textposition="top center")

    # Add 45-degree reference line
    max_x = float(d["Spend_share"].max() or 0)
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_x * 1.1,
        y1=max_x * 1.1,
        line=dict(dash="dash"),
    )

    fig.update_layout(
        xaxis_title="Spend share",
        yaxis_title="Engaged visits share",
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def heatmap_by_dimension_week(
    weekly_df: pd.DataFrame,
    dimension: str,
    metric: str,
    title: str,
) -> go.Figure:
    """Heatmap of metric by week and dimension (e.g. EVR by channel and week)."""
    if weekly_df.empty:
        return go.Figure()

    required = {dimension, "Week", metric}
    if not required.issubset(set(weekly_df.columns)):
        return go.Figure()

    pivot = (
        weekly_df.pivot_table(
            index=dimension,
            columns="Week",
            values=metric,
            aggfunc="mean",
        )
        .sort_index()
    )

    fig = px.imshow(
        pivot,
        aspect="auto",
        labels=dict(x="Week", y=dimension, color=metric),
        title=title,
        origin="lower",
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------
# Helpers for Campaign x Channel table + insights
# ---------------------------------------------------------------------

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _pct_change(curr, prev):
    """
    Vectorised + scalar-safe % change.
    Returns NaN when prev is 0 or missing.
    """
    curr = pd.to_numeric(curr, errors="coerce")
    prev = pd.to_numeric(prev, errors="coerce")

    if isinstance(prev, pd.Series):
        denom = prev.replace(0, np.nan)
    else:
        denom = np.nan if (pd.isna(prev) or prev == 0) else prev

    return (curr - prev) / denom


def _fmt_num(x, kind="number"):
    if pd.isna(x):
        return ""
    try:
        if kind == "pct":
            return f"{x * 100:.1f}%"
        if kind == "money":
            return f"£{x:,.0f}"
        if kind == "ratio":
            return f"{x:.3f}"
        return f"{x:,.0f}"
    except Exception:
        return str(x)


def _simple_insight(metric_key: str, pct: float) -> str:
    """
    Light-touch auto insight for the metrics table.
    Short + neutral.
    """
    if pd.isna(pct):
        return "No prior week"

    direction = "up" if pct > 0 else "down"
    magnitude = abs(pct)

    if metric_key in {"CPV", "CPE"}:
        if pct < -0.05:
            return f"Cost improved ({direction})"
        if pct > 0.05:
            return f"Cost worsened ({direction})"
        return "Cost stable"

    if magnitude >= 0.10:
        return f"Meaningful move ({direction})"
    if magnitude >= 0.05:
        return f"Moderate change ({direction})"
    return "Stable"


# ---------------------------------------------------------------------
# Campaign > Channel > Metrics HTML table
# ---------------------------------------------------------------------

def campaign_channel_metrics_html(
    df_raw: pd.DataFrame,
    metrics: Optional[List[Dict]] = None,
    value_col_label: Optional[str] = None,
    include_insights: bool = True,
) -> str:
    """
    Build a modern, grouped Campaign > Channel > Metrics table using HTML rowspans.

    Expects df_raw already normalised with:
      - Year, Quarter, Week
      - Campaign, Channel
      - Spend, Traffic, Engaged Visits (optional but recommended)
    """

    df = df_raw.copy()

    # Guard columns
    for col in ["Campaign", "Channel", "Year", "Quarter", "Week"]:
        if col not in df.columns:
            return (
                "<div style='padding:12px;border-radius:10px;background:#fff;"
                "border:1px solid #e5e7eb;'>"
                "<b>Campaign-channel table unavailable.</b><br/>"
                f"Missing column: <code>{col}</code>"
                "</div>"
            )

    # Default metric set aligned to your pipeline
    if metrics is None:
        metrics = [
            {"label": "Spend Pacing", "key": "Spend", "kind": "money"},
            {"label": "Visit Pacing", "key": "Traffic", "kind": "number"},
            {"label": "Engaged Visits", "key": "Engaged Visits", "kind": "number"},
            {"label": "EVR", "key": "EVR", "kind": "pct"},
            {"label": "CPV", "key": "CPV", "kind": "money"},
            {"label": "CPeV", "key": "CPE", "kind": "money"},
        ]

    # Aggregate weekly at Campaign x Channel
    agg = (
        df.groupby(["Year", "Quarter", "Week", "Campaign", "Channel"], dropna=False)
        .agg(
            Spend=("Spend", "sum") if "Spend" in df.columns else ("Campaign", "size"),
            Traffic=("Traffic", "sum") if "Traffic" in df.columns else ("Campaign", "size"),
            Engaged_Visits=("Engaged Visits", "sum") if "Engaged Visits" in df.columns else ("Campaign", "size"),
            Impressions=("Impressions", "sum") if "Impressions" in df.columns else ("Campaign", "size"),
            Clicks=("Clicks", "sum") if "Clicks" in df.columns else ("Campaign", "size"),
        )
        .reset_index()
    )

    # Coerce
    for c in ["Spend", "Traffic", "Engaged_Visits", "Impressions", "Clicks"]:
        if c in agg.columns:
            agg[c] = _to_num(agg[c])

    # Derived
    agg["EVR"] = np.where(agg["Traffic"] > 0, agg["Engaged_Visits"] / agg["Traffic"], np.nan)
    agg["CPV"] = np.where(agg["Traffic"] > 0, agg["Spend"] / agg["Traffic"], np.nan)
    agg["CPE"] = np.where(agg["Engaged_Visits"] > 0, agg["Spend"] / agg["Engaged_Visits"], np.nan)

    agg = agg.sort_values(["Year", "Quarter", "Week"])
    if agg.empty:
        return (
            "<div style='padding:12px;border-radius:10px;background:#fff;"
            "border:1px solid #e5e7eb;'>"
            "<b>No data available for this selection.</b>"
            "</div>"
        )

    # Latest week overall
    latest_row = agg.tail(1).iloc[0]
    latest_year = int(latest_row["Year"])
    latest_quarter = int(latest_row["Quarter"])
    latest_week = int(latest_row["Week"])

    latest_label = value_col_label or f"Q{latest_quarter}-W{latest_week}"

    # prev + pct by Campaign/Channel
    agg = agg.sort_values(["Campaign", "Channel", "Year", "Quarter", "Week"])
    for m in ["Spend", "Traffic", "Engaged_Visits", "EVR", "CPV", "CPE"]:
        agg[f"{m}_prev"] = agg.groupby(["Campaign", "Channel"], dropna=False)[m].shift(1)
        agg[f"{m}_pct"] = _pct_change(agg[m], agg[f"{m}_prev"])

    latest_df = agg[
        (agg["Year"] == latest_year) &
        (agg["Quarter"] == latest_quarter) &
        (agg["Week"] == latest_week)
    ].copy()

    if latest_df.empty:
        return (
            "<div style='padding:12px;border-radius:10px;background:#fff;"
            "border:1px solid #e5e7eb;'>"
            "<b>No latest-week data found for this selection.</b>"
            "</div>"
        )

    # Build long rows: Campaign, Channel, Metric, Value, %Change, Insight
    rows: List[Tuple[str, str, str, str, str, str]] = []

    for _, r in latest_df.iterrows():
        campaign = str(r["Campaign"])
        channel = str(r["Channel"])

        for spec in metrics:
            label = spec["label"]
            key = spec["key"]
            kind = spec.get("kind", "number")

            col_map = {"Engaged Visits": "Engaged_Visits"}
            col = col_map.get(key, key)

            val = r.get(col, np.nan)
            pct = r.get(f"{col}_pct", np.nan)

            money_keys = {"Spend", "CPV", "CPE"}
            val_kind = "money" if (kind == "money" or key in money_keys) else kind

            val_str = _fmt_num(val, kind=val_kind)
            pct_str = _fmt_num(pct, kind="pct")

            insight = _simple_insight(col if col in {"CPV", "CPE"} else key, pct) if include_insights else ""
            rows.append((campaign, channel, label, val_str, pct_str, insight))

    out = pd.DataFrame(rows, columns=["Campaign", "Channel", "Metric", latest_label, "%Change", "Insight"])
    out = out.sort_values(["Campaign", "Channel", "Metric"]).reset_index(drop=True)

    camp_counts = out.groupby("Campaign").size().to_dict()
    chan_counts = out.groupby(["Campaign", "Channel"]).size().to_dict()

    css = """
    <style>
      .ccm-wrap {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 14px 6px 14px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
      }
      table.ccm {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 13.5px;
      }
      .ccm thead th {
        text-align: left;
        background: #f8fafc;
        color: #0f172a;
        font-weight: 600;
        padding: 10px 12px;
        border-bottom: 1px solid #e5e7eb;
      }
      .ccm tbody td {
        padding: 9px 12px;
        border-bottom: 1px solid #f1f5f9;
        vertical-align: top;
      }
      .ccm tbody tr:last-child td {
        border-bottom: none;
      }
      .ccm .muted {
        color: #64748b;
        font-weight: 500;
      }
      .ccm .metric {
        color: #0f172a;
        font-weight: 500;
      }
      .ccm .pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: #f1f5f9;
        color: #0f172a;
        font-size: 11px;
        font-weight: 600;
      }
      .ccm .insight {
        color: #334155;
        font-size: 12px;
      }
      .ccm .right {
        text-align: right;
      }
    </style>
    """

    html = [css, "<div class='ccm-wrap'>"]
    html.append("<table class='ccm'>")
    html.append("<thead><tr>")
    html.append("<th>Campaign</th>")
    html.append("<th>Channel</th>")
    html.append("<th>Focus Metrics</th>")
    html.append(f"<th class='right'>{latest_label}</th>")
    html.append("<th class='right'>%Change</th>")
    if include_insights:
        html.append("<th>Commentary</th>")
    html.append("</tr></thead>")
    html.append("<tbody>")

    seen_campaign = set()
    seen_channel = set()

    for _, row in out.iterrows():
        camp = row["Campaign"]
        chan = row["Channel"]

        html.append("<tr>")

        if camp not in seen_campaign:
            rowspan = int(camp_counts.get(camp, 1))
            html.append(f"<td rowspan='{rowspan}'><span class='pill'>{camp}</span></td>")
            seen_campaign.add(camp)

        chan_key = (camp, chan)
        if chan_key not in seen_channel:
            rowspan = int(chan_counts.get(chan_key, 1))
            html.append(f"<td rowspan='{rowspan}' class='muted'>{chan}</td>")
            seen_channel.add(chan_key)

        html.append(f"<td class='metric'>{row['Metric']}</td>")
        html.append(f"<td class='right'>{row[latest_label]}</td>")
        html.append(f"<td class='right'>{row['%Change']}</td>")

        if include_insights:
            html.append(f"<td class='insight'>{row['Insight']}</td>")

        html.append("</tr>")

    html.append("</tbody></table></div>")
    return "".join(html)


# ---------------------------------------------------------------------
# Campaign x Channel copy-ready insights (no optimisation language)
# ---------------------------------------------------------------------

def campaign_channel_insights(
    df_raw: pd.DataFrame,
    max_bullets: int = 3,
) -> Dict[Tuple[str, str], List[str]]:
    """
    Returns paste-ready insights keyed by (Campaign, Channel).
    Uses latest-week values + WoW % + simple share context.
    Avoids optimisation language.
    """

    df = df_raw.copy()

    required = {"Campaign", "Channel", "Year", "Quarter", "Week"}
    if not required.issubset(set(df.columns)):
        return {}

    agg = (
        df.groupby(["Year", "Quarter", "Week", "Campaign", "Channel"], dropna=False)
        .agg(
            Spend=("Spend", "sum") if "Spend" in df.columns else ("Campaign", "size"),
            Traffic=("Traffic", "sum") if "Traffic" in df.columns else ("Campaign", "size"),
            Engaged_Visits=("Engaged Visits", "sum") if "Engaged Visits" in df.columns else ("Campaign", "size"),
        )
        .reset_index()
    )

    for c in ["Spend", "Traffic", "Engaged_Visits"]:
        if c in agg.columns:
            agg[c] = pd.to_numeric(agg[c], errors="coerce")

    agg["EVR"] = np.where(agg["Traffic"] > 0, agg["Engaged_Visits"] / agg["Traffic"], np.nan)
    agg["CPV"] = np.where(agg["Traffic"] > 0, agg["Spend"] / agg["Traffic"], np.nan)
    agg["CPE"] = np.where(agg["Engaged_Visits"] > 0, agg["Spend"] / agg["Engaged_Visits"], np.nan)

    agg = agg.sort_values(["Year", "Quarter", "Week"])
    if agg.empty:
        return {}

    latest_row = agg.tail(1).iloc[0]
    ly, lq, lw = int(latest_row["Year"]), int(latest_row["Quarter"]), int(latest_row["Week"])

    agg = agg.sort_values(["Campaign", "Channel", "Year", "Quarter", "Week"])
    for m in ["Spend", "Traffic", "Engaged_Visits", "EVR", "CPV", "CPE"]:
        agg[f"{m}_prev"] = agg.groupby(["Campaign", "Channel"], dropna=False)[m].shift(1)
        agg[f"{m}_pct"] = _pct_change(agg[m], agg[f"{m}_prev"])

    latest_df = agg[
        (agg["Year"] == ly) & (agg["Quarter"] == lq) & (agg["Week"] == lw)
    ].copy()

    if latest_df.empty:
        return {}

    camp_totals = (
        latest_df.groupby("Campaign", dropna=False)
        .agg(
            Spend_total=("Spend", "sum"),
            Eng_total=("Engaged_Visits", "sum"),
        )
        .reset_index()
    )

    latest_df = latest_df.merge(camp_totals, on="Campaign", how="left")
    latest_df["Spend_share"] = latest_df["Spend"] / latest_df["Spend_total"].replace(0, np.nan)
    latest_df["Eng_share"] = latest_df["Engaged_Visits"] / latest_df["Eng_total"].replace(0, np.nan)

    def _dir(p):
        if pd.isna(p):
            return None
        return "up" if p > 0 else "down"

    def _big_move(p, t=0.10):
        return (not pd.isna(p)) and (abs(p) >= t)

    insights: Dict[Tuple[str, str], List[str]] = {}

    for _, r in latest_df.iterrows():
        camp = str(r["Campaign"])
        chan = str(r["Channel"])
        key = (camp, chan)

        bullets: List[str] = []

        # 1) Contribution statement
        if not pd.isna(r.get("Eng_share")):
            es = float(r["Eng_share"]) * 100
            bullets.append(f"Contributes {es:.1f}% of this campaign’s engaged visits this week.")

        # 2) EVR statement
        evr = r.get("EVR", np.nan)
        evr_pct = r.get("EVR_pct", np.nan)
        if not pd.isna(evr):
            if _big_move(evr_pct, 0.08):
                bullets.append(f"Engagement rate is {evr*100:.1f}% and moved {_dir(evr_pct)} WoW.")
            else:
                bullets.append(f"Engagement rate sits at {evr*100:.1f}%.")

        # 3) CPE statement
        cpe = r.get("CPE", np.nan)
        cpe_pct = r.get("CPE_pct", np.nan)
        if not pd.isna(cpe):
            if _big_move(cpe_pct, 0.10):
                bullets.append(f"Cost per engaged visit is £{cpe:,.0f} and moved {_dir(cpe_pct)} WoW.")
            else:
                bullets.append(f"Cost per engaged visit is £{cpe:,.0f}.")

        # 4) Add spend/traffic movement only if needed for variety
        sp_pct = r.get("Spend_pct", np.nan)
        tr_pct = r.get("Traffic_pct", np.nan)

        if len(bullets) < max_bullets:
            if _big_move(sp_pct, 0.12):
                bullets.append(f"Spend moved {_dir(sp_pct)} WoW.")
            elif _big_move(tr_pct, 0.12):
                bullets.append(f"Traffic moved {_dir(tr_pct)} WoW.")

        # Dedupe + trim
        cleaned: List[str] = []
        for b in bullets:
            if b and b not in cleaned:
                cleaned.append(b)

        insights[key] = cleaned[:max_bullets]

    return insights

def campaign_channel_insights_text_blocks(
    df_raw: pd.DataFrame,
    max_bullets: int = 3,
) -> Dict[str, str]:
    """
    Returns {campaign_name: multi-line text block}
    with channel subheaders and bullets.
    Perfect for Streamlit text_area copy boxes.
    """
    insights = campaign_channel_insights(df_raw, max_bullets=max_bullets)
    if not insights:
        return {}

    # group by campaign
    grouped: Dict[str, Dict[str, List[str]]] = {}
    for (camp, chan), bullets in insights.items():
        grouped.setdefault(camp, {})[chan] = bullets

    out: Dict[str, str] = {}

    for camp, chans in grouped.items():
        lines: List[str] = []
        lines.append(f"{camp} — latest week highlights")
        for chan, bullets in chans.items():
            lines.append(f"\n{chan}")
            for b in bullets:
                lines.append(f"• {b}")
        out[camp] = "\n".join(lines).strip()

    return out


def campaign_channel_insights_html(
    df_raw: pd.DataFrame,
    max_bullets: int = 3,
) -> str:
    """
    Renders a clean copy-ready insight block grouped by Campaign > Channel.
    """

    insights = campaign_channel_insights(df_raw, max_bullets=max_bullets)
    if not insights:
        return (
            "<div style='padding:12px;border-radius:10px;background:#fff;"
            "border:1px solid #e5e7eb;'>"
            "<b>No campaign/channel insights available for this selection.</b>"
            "</div>"
        )

    grouped: Dict[str, Dict[str, List[str]]] = {}
    for (camp, chan), bullets in insights.items():
        grouped.setdefault(camp, {})[chan] = bullets

    css = """
    <style>
      .cci-wrap{
        background:#ffffff;
        border:1px solid #e5e7eb;
        border-radius:14px;
        padding:16px;
        box-shadow:0 8px 20px rgba(15,23,42,0.04);
      }
      .cci-camp{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-weight:700;
        font-size:16px;
        color:#0f172a;
        margin: 8px 0 10px 0;
      }
      .cci-chan{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-weight:600;
        font-size:14px;
        color:#334155;
        margin: 10px 0 6px 0;
      }
      .cci-ul{
        margin: 0 0 10px 18px;
        color:#0f172a;
        font-size:13.5px;
      }
      .cci-li{
        margin-bottom:4px;
      }
      .cci-pill{
        display:inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background:#f1f5f9;
        color:#0f172a;
        font-size:11px;
        font-weight:600;
        margin-right:6px;
      }
    </style>
    """

    html = [css, "<div class='cci-wrap'>"]

    for camp, chans in grouped.items():
        html.append(f"<div class='cci-camp'><span class='cci-pill'>Campaign</span>{camp}</div>")
        for chan, bullets in chans.items():
            html.append(f"<div class='cci-chan'>{chan}</div>")
            html.append("<ul class='cci-ul'>")
            for b in bullets:
                html.append(f"<li class='cci-li'>{b}</li>")
            html.append("</ul>")

    html.append("</div>")
    return "".join(html)
