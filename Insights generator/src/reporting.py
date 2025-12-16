from typing import Dict, List
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.io as pio

from .ml_models import feature_importance_engagement
from .config import INSIGHT_THRESHOLDS as T


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x * 100:.1f}%"


def _fmt_num(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f}k"
    return f"{x:.0f}"


def _fmt_int(x) -> str:
    try:
        if pd.isna(x):
            return "?"
        return str(int(x))
    except Exception:
        return "?"


def build_headline_insights(overall_weekly: pd.DataFrame) -> List[str]:
    """High level latest week view with a proper diagnostic of what drove the change."""
    lines: List[str] = []
    if overall_weekly.empty:
        return ["No data available after filtering."]

    df = overall_weekly.sort_values(["Year", "Quarter", "Week"]).copy()
    latest_two = df.tail(2)

    # If there is only one week, keep it simple
    if len(latest_two) == 1:
        curr = latest_two.iloc[0]
        lines.append(
            f"Q{_fmt_int(curr.get('Quarter'))} week {_fmt_int(curr.get('Week'))} spend is £{_fmt_num(curr['Spend'])}, "
            f"traffic {_fmt_num(curr['Traffic'])} visits, engaged visits {_fmt_num(curr['Engaged_Visits'])} "
            f"with EVR {_fmt_pct(curr['EVR'])}."
        )
        return lines

    prev, curr = latest_two.iloc[0], latest_two.iloc[1]

    def safe_pct_change(now, then):
        if pd.isna(now) or pd.isna(then) or then == 0:
            return np.nan
        return (now - then) / then

    spend_wow = safe_pct_change(curr["Spend"], prev["Spend"])
    traf_wow = safe_pct_change(curr["Traffic"], prev["Traffic"])
    evr_wow = safe_pct_change(curr["EVR"], prev["EVR"])

    # 1. Core headline with WoW movement
    lines.append(
        f"Latest week Q{_fmt_int(curr.get('Quarter'))} W{_fmt_int(curr.get('Week'))} spend is £{_fmt_num(curr['Spend'])} "
        f"({_fmt_pct(spend_wow)} vs last week), traffic {_fmt_num(curr['Traffic'])} "
        f"({_fmt_pct(traf_wow)} WoW) and EVR {_fmt_pct(curr['EVR'])} "
        f"({_fmt_pct(evr_wow)} WoW)."
    )

    # 2. Decompose change in engaged visits into scale vs quality
    prev_ev = float(prev.get("Engaged_Visits", np.nan))
    curr_ev = float(curr.get("Engaged_Visits", np.nan))
    if not pd.isna(prev_ev) and not pd.isna(curr_ev):
        delta_ev = curr_ev - prev_ev

        prev_traf = float(prev.get("Traffic", np.nan))
        curr_traf = float(curr.get("Traffic", np.nan))
        prev_evr = float(prev.get("EVR", np.nan))
        curr_evr = float(curr.get("EVR", np.nan))

        if all(not pd.isna(v) for v in [prev_traf, curr_traf, prev_evr, curr_evr]):
            d_traf = curr_traf - prev_traf
            d_evr = curr_evr - prev_evr

            scale_effect = d_traf * prev_evr
            rate_effect = prev_traf * d_evr
            interaction = d_traf * d_evr
            total_modelled = scale_effect + rate_effect + interaction or 1.0

            contrib_scale = scale_effect / total_modelled
            contrib_rate = rate_effect / total_modelled

            direction = "up" if delta_ev > 0 else "down"
            lines.append(
                f"Engaged visits moved {direction} by {_fmt_num(abs(delta_ev))} vs last week. "
                f"Around {_fmt_pct(contrib_scale)} of this shift is explained by changes in traffic volume, "
                f"and {_fmt_pct(contrib_rate)} by changes in visit quality (EVR)."
            )

    # 3. Compare latest week EVR and CPE to recent run rate
    lookback = df.tail(min(6, len(df)))  # last six weeks if possible
    baseline = lookback.head(len(lookback) - 1)  # exclude current week
    if not baseline.empty:
        def avg(series):
            return series.replace([np.inf, -np.inf], np.nan).mean()

        base_evr = avg(baseline["EVR"])
        base_cpe = avg(baseline["CPE"]) if "CPE" in baseline.columns else np.nan

        evr_vs_baseline = safe_pct_change(curr["EVR"], base_evr) if not pd.isna(base_evr) else np.nan
        cpe_vs_baseline = safe_pct_change(curr.get("CPE", np.nan), base_cpe) if not pd.isna(base_cpe) else np.nan

        if not pd.isna(evr_vs_baseline) and abs(evr_vs_baseline) >= T.min_rel_change_to_comment:
            move = "above" if evr_vs_baseline > 0 else "below"
            lines.append(
                f"EVR this week sits {move} the recent average by {_fmt_pct(abs(evr_vs_baseline))}."
            )

        if not pd.isna(cpe_vs_baseline) and abs(cpe_vs_baseline) >= T.min_rel_change_to_comment:
            move = "higher" if cpe_vs_baseline > 0 else "lower"
            lines.append(
                f"Cost per engaged visit is {move} than the recent average by {_fmt_pct(abs(cpe_vs_baseline))}, "
                f"suggesting a {'less efficient' if cpe_vs_baseline > 0 else 'more efficient'} week in cost terms."
            )

    # 4. Simple EV vs run rate using a rolling mean
    window = min(6, len(df))
    df["EV_roll_mean"] = df["Engaged_Visits"].replace([np.inf, -np.inf], np.nan).rolling(
        window=window, min_periods=3
    ).mean()
    expected_ev = df["EV_roll_mean"].iloc[-1]

    if not pd.isna(expected_ev) and expected_ev > 0:
        ev_diff = safe_pct_change(curr_ev, expected_ev)
        if not pd.isna(ev_diff) and abs(ev_diff) >= T.min_rel_change_to_comment:
            direction = "above" if ev_diff > 0 else "below"
            lines.append(
                f"Engaged visits sit {direction} the recent {window}-week run rate by {_fmt_pct(abs(ev_diff))}."
            )

    return lines



def build_anomaly_insights(overall_weekly: pd.DataFrame) -> List[str]:
    """Surface Isolation Forest anomalies as narrative insights."""
    lines: List[str] = []
    if overall_weekly.empty or "is_anomaly" not in overall_weekly.columns:
        return lines

    df = overall_weekly.sort_values(["Year", "Quarter", "Week"]).copy()
    anomalies = df[df["is_anomaly"].astype(bool)]

    if anomalies.empty:
        lines.append(
            "No weeks in the current period are flagged as statistically unusual by the anomaly model."
        )
        return lines

    described = set()

    # 1. Call out if the latest week is actually anomalous
    latest = df.iloc[-1]
    if bool(latest.get("is_anomaly", False)):
        prev = df.iloc[-2] if len(df) >= 2 else None
        contrib_text = ""
        if prev is not None:
            diffs = {}
            for m in ["Spend", "Traffic", "Engaged_Visits"]:
                if m in df.columns:
                    base = prev[m]
                    curr = latest[m]
                    if base not in (0, None) and not pd.isna(base) and not pd.isna(curr):
                        diffs[m] = (curr - base) / base
            if diffs:
                top_metric, top_val = max(diffs.items(), key=lambda x: abs(x[1]))
                direction = "up" if top_val > 0 else "down"
                contrib_text = (
                    f" mainly because {top_metric.lower()} moves {direction} by "
                    f"{_fmt_pct(top_val)} versus last week"
                )

        lines.append(
            f"The latest week Q{int(latest['Quarter'])} W{int(latest['Week'])} "
            f"is flagged as an anomaly by the Isolation Forest model{contrib_text}."
        )
        described.add((int(latest["Quarter"]), int(latest["Week"])))

    # 2. Also highlight the most extreme anomalies in the period
    if "anomaly_score" in df.columns:
        extremes = anomalies.nsmallest(2, "anomaly_score")  # most negative = most anomalous
    else:
        extremes = anomalies.tail(2)

    for _, row in extremes.iterrows():
        key = (int(row["Quarter"]), int(row["Week"]))
        if key in described:
            continue
        lines.append(
            f"Week Q{int(row['Quarter'])} W{int(row['Week'])} shows an unusual mix of spend "
            f"£{_fmt_num(row['Spend'])}, traffic {_fmt_num(row['Traffic'])} and engaged visits "
            f"{_fmt_num(row['Engaged_Visits'])} compared with the rest of the period."
        )
        described.add(key)

    return lines


def build_channel_insights(channel_qtd: pd.DataFrame) -> List[str]:
    """What channels are doing in volume and efficiency terms."""
    lines: List[str] = []
    if channel_qtd.empty:
        return lines

    d = channel_qtd.copy()
    total_spend = d["Spend_QTD"].sum() or 1.0
    total_eng = d["Engaged_Visits_QTD"].sum() or 1.0

    d["Spend_share"] = d["Spend_QTD"] / total_spend
    d["Eng_share"] = d["Engaged_Visits_QTD"] / total_eng
    d = d.sort_values("Spend_QTD", ascending=False)

    top_channels = d.head(5).to_dict("records")

    for row in top_channels:
        lines.append(
            f"{row['Channel']} delivers {_fmt_pct(row['Eng_share'])} of engaged visits from "
            f"{_fmt_pct(row['Spend_share'])} of spend (EVR {_fmt_pct(row['EVR_QTD'])}, "
            f"CPE £{_fmt_num(row['CPE_QTD'])})."
        )

    best = d.nsmallest(1, "CPE_QTD").iloc[0]
    worst = d.nlargest(1, "CPE_QTD").iloc[0]
    lines.append(
        f"Most efficient channel by CPE is {best['Channel']} at £{_fmt_num(best['CPE_QTD'])} "
        f"per engaged visit, while {worst['Channel']} is the most expensive at "
        f"£{_fmt_num(worst['CPE_QTD'])}."
    )

    return lines


def build_market_insights(market_qtd: pd.DataFrame) -> List[str]:
    """Volume and efficiency by market with hotspots and risk areas vs portfolio averages."""
    lines: List[str] = []
    if market_qtd.empty:
        return lines

    d = market_qtd.copy()
    if not {"Engaged_Visits_QTD", "EVR_QTD", "CPE_QTD"}.issubset(d.columns):
        # Fallback. keep something rather than failing silently
        d = d.sort_values("Engaged_Visits_QTD", ascending=False)
        for _, row in d.head(5).iterrows():
            lines.append(
                f"{row['Market']} delivers {_fmt_num(row['Engaged_Visits_QTD'])} engaged visits QTD "
                f"with EVR {_fmt_pct(row['EVR_QTD'])} and CPE £{_fmt_num(row['CPE_QTD'])}."
            )
        return lines

    d = d.sort_values("Engaged_Visits_QTD", ascending=False)

    # Portfolio benchmarks
    overall_evr = d["EVR_QTD"].replace([np.inf, -np.inf], np.nan).mean()
    overall_cpe = d["CPE_QTD"].replace([np.inf, -np.inf], np.nan).mean()

    # 1. Core volume and efficiency for top markets
    for _, row in d.head(5).iterrows():
        lines.append(
            f"{row['Market']} delivers {_fmt_num(row['Engaged_Visits_QTD'])} engaged visits QTD "
            f"with EVR {_fmt_pct(row['EVR_QTD'])} and CPE £{_fmt_num(row['CPE_QTD'])}."
        )

    # 2. Markets significantly ahead or behind the portfolio
    if not pd.isna(overall_evr) and not pd.isna(overall_cpe):
        hot = d[
            (d["EVR_QTD"] >= overall_evr * (1 + T.evr_hot_uplift))
            & (d["CPE_QTD"] <= overall_cpe * (1 - T.cpe_hot_saving))
        ]
        cold = d[
            (d["EVR_QTD"] <= overall_evr * (1 - T.evr_cold_drop))
            | (d["CPE_QTD"] >= overall_cpe * (1 + T.cpe_cold_penalty))
        ]

        if not hot.empty:
            parts = []
            for _, row in hot.head(3).iterrows():
                evr_uplift = (row["EVR_QTD"] / overall_evr) - 1 if overall_evr else np.nan
                cpe_saving = (overall_cpe / row["CPE_QTD"]) - 1 if row["CPE_QTD"] else np.nan
                parts.append(
                    f"{row['Market']} (EVR {_fmt_pct(row['EVR_QTD'])}, "
                    f"around {_fmt_pct(evr_uplift)} above average and CPE around {_fmt_pct(cpe_saving)} cheaper)"
                )
            lines.append(
                "High performing markets that pull the portfolio up on both rate and cost include "
                + ", ".join(parts)
                + "."
            )

        if not cold.empty:
            parts = []
            for _, row in cold.head(3).iterrows():
                evr_gap = (overall_evr / row["EVR_QTD"]) - 1 if row["EVR_QTD"] else np.nan
                cpe_penalty = (row["CPE_QTD"] / overall_cpe) - 1 if overall_cpe else np.nan
                parts.append(
                    f"{row['Market']} (EVR {_fmt_pct(row['EVR_QTD'])}, "
                    f"around {_fmt_pct(evr_gap)} below average and CPE around {_fmt_pct(cpe_penalty)} higher)"
                )
            lines.append(
                "Markets acting as a drag on overall efficiency include "
                + ", ".join(parts)
                + ". These should be prioritised for diagnostic work on creative fit, local journey and targeting."
            )

    return lines


def build_seasonality_insights(overall_weekly: pd.DataFrame) -> List[str]:
    """Comment on intra quarter trend and basic relationships (seasonality style)."""
    lines: List[str] = []
    if overall_weekly.empty or len(overall_weekly) < 3:
        return lines

    df = overall_weekly.sort_values(["Year", "Quarter", "Week"]).copy()

    # Early vs late in period
    n = len(df)
    first = df.head(max(1, n // 2))
    last = df.tail(max(1, n // 2))

    def _avg(series):
        return series.replace([np.inf, -np.inf], np.nan).mean()

    spend_first, spend_last = _avg(first["Spend"]), _avg(last["Spend"])
    evr_first, evr_last = _avg(first["EVR"]), _avg(last["EVR"])
    traf_first, traf_last = _avg(first["Traffic"]), _avg(last["Traffic"])

    def _delta_desc(label: str, v1, v2, unit: str = "") -> str:
        if pd.isna(v1) or pd.isna(v2) or v1 == 0:
            return ""
        diff = v2 - v1
        pct = diff / v1
        if abs(pct) < 0.05:  # less than 5 percent change => fairly flat
            return f"{label} is broadly flat across the period."
        direction = "increases" if diff > 0 else "softens"
        sign = "+" if diff > 0 else ""
        if unit == "£":
            return f"{label} {direction} from {unit}{_fmt_num(v1)} to {unit}{_fmt_num(v2)} ({sign}{_fmt_pct(pct)})."
        if unit == "%":
            return f"{label} {direction} from {_fmt_pct(v1)} to {_fmt_pct(v2)} ({sign}{_fmt_pct(pct)})."
        return f"{label} {direction} from {_fmt_num(v1)} to {_fmt_num(v2)} ({sign}{_fmt_pct(pct)})."

    desc_spend = _delta_desc("Average weekly spend", spend_first, spend_last, unit="£")
    desc_traf = _delta_desc("Average weekly traffic", traf_first, traf_last)
    desc_evr = _delta_desc("Average weekly engagement rate", evr_first, evr_last, unit="%")

    for d in [desc_spend, desc_traf, desc_evr]:
        if d:
            lines.append(d)

    # Correlations between variables (gives more of the 'why' flavour)
    if df["Spend"].nunique() > 2 and df["EVR"].nunique() > 2:
        corr = df["Spend"].corr(df["EVR"])
        if not pd.isna(corr) and abs(corr) >= T.corr_min_abs:
            if corr > 0:
                lines.append(
                    "Weeks with higher spend tend to see higher engagement rates, suggesting more investment is associated with better quality traffic."
                )
            else:
                lines.append(
                    "Weeks with higher spend tend to see lower engagement rates, suggesting that incremental spend may be reaching less engaged audiences or less effective environments."
                )

    if df["Traffic"].nunique() > 2 and df["EVR"].nunique() > 2:
        corr = df["Traffic"].corr(df["EVR"])
        if not pd.isna(corr) and abs(corr) >= 0.3:
            if corr > 0:
                lines.append(
                    "Traffic and EVR move together, so scale and quality of visits are aligned in this period."
                )
            else:
                lines.append(
                    "Traffic and EVR move in opposite directions, hinting at a volume versus quality trade off as more visits are driven."
                )

    return lines


def build_mix_efficiency_insights(channel_qtd: pd.DataFrame) -> List[str]:
    """Classify channels into over and under delivering and quantify a simple reallocation upside."""
    lines: List[str] = []
    if channel_qtd.empty:
        return lines

    d = channel_qtd.copy()
    if not {"Spend_QTD", "Engaged_Visits_QTD"}.issubset(d.columns):
        return lines

    total_spend = d["Spend_QTD"].sum() or 1.0
    total_eng = d["Engaged_Visits_QTD"].sum() or 1.0
    d["Spend_share"] = d["Spend_QTD"] / total_spend
    d["Eng_share"] = d["Engaged_Visits_QTD"] / total_eng

    # Identify clear over and under performers
    over = d[d["Eng_share"] >= d["Spend_share"] + T.share_gap_strong]
    under = d[d["Spend_share"] >= d["Eng_share"] + T.share_gap_strong]

    if not over.empty:
        top_over = ", ".join(
            f"{row['Channel']} (EV share {_fmt_pct(row['Eng_share'])} vs spend share {_fmt_pct(row['Spend_share'])})"
            for _, row in over.head(3).iterrows()
        )
        lines.append(
            f"Over delivering channels on engagement include {top_over}. "
            f"These are strong candidates for incremental budget while performance remains ahead of investment."
        )

    if not under.empty:
        top_under = ", ".join(
            f"{row['Channel']} (spend share {_fmt_pct(row['Spend_share'])} vs EV share {_fmt_pct(row['Eng_share'])})"
            for _, row in under.head(3).iterrows()
        )
        lines.append(
            f"Channels absorbing more budget than they return in engaged visits include {top_under}. "
            f"These should be challenged on role and creative approach or used more selectively."
        )

    # Simple what if reallocation from the weakest to the strongest channel
    if not over.empty and not under.empty and "CPE_QTD" in d.columns:
        best = d.loc[over.index].sort_values("CPE_QTD").iloc[0]
        worst = d.loc[under.index].sort_values("CPE_QTD", ascending=False).iloc[0]

        shift_share = 0.10
        shift_spend = total_spend * shift_share

        if best["CPE_QTD"] > 0 and worst["CPE_QTD"] > 0:
            ev_from_best = shift_spend / best["CPE_QTD"]
            ev_from_worst = shift_spend / worst["CPE_QTD"]
            incremental_ev = ev_from_best - ev_from_worst

            if incremental_ev > 0:
                lines.append(
                    f"If {shift_share:.0%} of total QTD spend shifted from {worst['Channel']} to {best['Channel']} at current CPE levels, "
                    f"the plan would deliver roughly {_fmt_num(incremental_ev)} additional engaged visits, "
                    f"assuming response remains stable."
                )

    return lines


def build_contributor_insights(
    weekly_df: pd.DataFrame,
    dim: str = "Channel",
    metric: str = "Engaged_Visits",
    top_n: int = 3,
) -> List[str]:
    """
    Identify which members of a dimension explain most of the WoW change in a metric.

    This is fully generic.
    It works for Channel, Market, Device etc as long as the weekly_df has those columns.
    """
    lines: List[str] = []
    if weekly_df.empty or dim not in weekly_df.columns or metric not in weekly_df.columns:
        return lines

    df = weekly_df.sort_values(["Year", "Quarter", "Week"])
    if df["Week"].nunique() < 2:
        return lines

    # Aggregate to one row per dim per week
    last_two = (
        df.groupby([dim, "Year", "Quarter", "Week"], dropna=False)[metric]
        .sum()
        .reset_index()
    )
    latest_week = last_two["Week"].max()
    prev_week = sorted(last_two["Week"].unique())[-2]

    curr = last_two[last_two["Week"] == latest_week]
    prev = last_two[last_two["Week"] == prev_week]

    merged = curr.merge(prev, on=dim, how="outer", suffixes=("_curr", "_prev")).fillna(0.0)
    merged["delta"] = merged[f"{metric}_curr"] - merged[f"{metric}_prev"]

    total_delta = merged["delta"].sum()
    if total_delta == 0:
        return lines

    inc = merged[merged["delta"] > 0].sort_values("delta", ascending=False).head(top_n)
    dec = merged[merged["delta"] < 0].sort_values("delta").head(top_n)

    if not inc.empty:
        share = inc["delta"].sum() / total_delta if total_delta != 0 else np.nan
        names = ", ".join(
            f"{row[dim]} (+{_fmt_num(row['delta'])})"
            for _, row in inc.iterrows()
        )
        lines.append(
            f"On {dim.lower()} level, the main contributors to the week on week uplift in "
            f"{metric.replace('_', ' ').lower()} are {names}, together explaining {_fmt_pct(share)} of the overall change."
        )

    if not dec.empty:
        neg_total = dec["delta"].sum()
        share = neg_total / total_delta if total_delta != 0 else np.nan
        names = ", ".join(
            f"{row[dim]} ({_fmt_num(row['delta'])})"
            for _, row in dec.iterrows()
        )
        lines.append(
            f"The biggest drags are {names}, which together account for {_fmt_pct(abs(share))} of the movement."
        )

    return lines


def _volatility_summary(weekly_df: pd.DataFrame, dim: str, metric: str, min_weeks: int = 3) -> List[str]:
    lines: List[str] = []
    if weekly_df.empty or dim not in weekly_df.columns:
        return lines

    grp = (
        weekly_df.groupby(dim, dropna=False)
        .agg(
            mean_val=(metric, "mean"),
            std_val=(metric, "std"),
            weeks=("Week", "nunique"),
        )
        .reset_index()
    )

    grp = grp[(grp["weeks"] >= min_weeks) & (grp["mean_val"] > 0)]
    if grp.empty:
        return lines

    grp["cv"] = grp["std_val"] / grp["mean_val"]
    top = grp.sort_values("cv", ascending=False).head(3)

    for _, row in top.iterrows():
        lines.append(
            f"{row[dim]} shows high week on week volatility in {metric.lower()} (coefficient of variation {row['cv']:.2f}), "
            f"meaning results swing more than average from week to week."
        )

    return lines


def build_volatility_insights(channel_weekly: pd.DataFrame, market_weekly: pd.DataFrame) -> List[str]:
    """Highlight channels / markets with unstable delivery over time."""
    lines: List[str] = []
    lines.extend(_volatility_summary(channel_weekly, "Channel", "Spend"))
    lines.extend(_volatility_summary(channel_weekly, "Channel", "Engaged_Visits"))
    if not lines:
        lines.extend(_volatility_summary(market_weekly, "Market", "Spend"))
    return lines


def build_driver_insights(df_filtered: pd.DataFrame) -> List[str]:
    """Feature importance based narrative with clearer focus on what matters most."""
    lines: List[str] = []
    drivers = feature_importance_engagement(df_filtered)

    if not drivers:
        return lines

    sorted_drivers = sorted(drivers.items(), key=lambda x: x[1], reverse=True)
    top = sorted_drivers[:5]

    parts = [f"{name} ({score * 100:.0f}%)" for name, score in top]
    lines.append(
        "Model based driver analysis suggests that engaged visits are most strongly associated with: "
        + ", ".join(parts)
        + "."
    )

    total_top2 = sum(score for _, score in top[:2])
    if total_top2 >= 0.5:
        key_dims = ", ".join(name for name, _ in top[:2])
        lines.append(
            f"The top two drivers {key_dims} alone explain around {_fmt_pct(total_top2)} of the modelled variation, "
            f"so testing and budget decisions should focus on these first."
        )

    return lines


def build_html_report(
    output_path: Path,
    figures: Dict[str, "plotly.graph_objects.Figure"],
    text_blocks: Dict[str, List[str]],
) -> None:
    """HTML report used by the CLI pathway."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_html_parts = []
    first = True
    for name, fig in figures.items():
        fragment = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs="cdn" if first else False,
            default_height="420px",
        )
        fig_html_parts.append(f"<h3>{name}</h3>\n{fragment}")
        first = False

    def block_html(title: str, lines: List[str]) -> str:
        items = "".join(f"<li>{line}</li>" for line in lines)
        return f"<section><h2>{title}</h2><ul>{items}</ul></section>"

    sections_html = []
    sections_html.append(block_html("Headline performance", text_blocks.get("headline", [])))
    sections_html.append(block_html("Anomalies and spikes", text_blocks.get("anomalies", [])))
    sections_html.append(block_html("Seasonality and trends", text_blocks.get("seasonality", [])))
    sections_html.append(block_html("Channel performance", text_blocks.get("channel", [])))
    sections_html.append(block_html("Channel level movers", text_blocks.get("contributors_channel", [])))
    sections_html.append(block_html("Market performance", text_blocks.get("market", [])))
    sections_html.append(block_html("Market level movers", text_blocks.get("contributors_market", [])))
    sections_html.append(block_html("Mix and efficiency", text_blocks.get("mix", [])))
    sections_html.append(block_html("Volatility and stability", text_blocks.get("volatility", [])))
    sections_html.append(block_html("Drivers and modelling", text_blocks.get("drivers", [])))

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Automated media insights</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 0;
            padding: 2rem 3rem;
            background: #fafafa;
            color: #222;
        }}
        h1, h2, h3 {{
            font-weight: 600;
        }}
        section {{
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
        }}
        ul {{
            padding-left: 1.25rem;
        }}
        li {{
            margin-bottom: 0.4rem;
            line-height: 1.4;
        }}
        .figures {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            grid-gap: 1.5rem;
            margin-top: 1rem;
        }}
        .figure-card {{
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
            padding: 1rem;
        }}
        .figure-card h3 {{
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <h1>Automated media insights</h1>
    {''.join(sections_html)}
    <section>
        <h2>Visual summary</h2>
        <div class="figures">
            {''.join(f'<div class="figure-card">{frag}</div>' for frag in fig_html_parts)}
        </div>
    </section>
</body>
</html>
    """

    output_path.write_text(html, encoding="utf-8")

# ============================================================
# NEW FUNCTIONS ADDED FOR POWERPOINT CAMPAIGN SLIDE BUILDER
# ============================================================

def compute_campaign_kpis(df_campaign):
    """
    Computes KPIs for the latest week of a campaign.
    Returns:
        {
            "Spend": {"value": X, "delta": Y},
            "Traffic": {"value": X, "delta": Y},
            "Engaged_Visits": {"value": X, "delta": Y},
            "EVR": {"value": X, "delta": Y},
        }
    """
    if df_campaign.empty:
        return {}

    df = df_campaign.sort_values(["Year", "Quarter", "Week"]).copy()

    latest = df.tail(1)
    prev = df.tail(2).head(1) if len(df) >= 2 else None

    # derive EVR if missing
    if "EVR" not in df.columns and {"Traffic", "Engaged Visits"}.issubset(df.columns):
        df["EVR"] = np.where(df["Traffic"] > 0, df["Engaged Visits"] / df["Traffic"], np.nan)
        latest = df.tail(1)
        prev = df.tail(2).head(1) if len(df) >= 2 else None

    def extract(col):
        val = latest[col].iloc[0] if col in latest.columns else np.nan
        if prev is None or col not in prev.columns:
            return {"value": val, "delta": np.nan}

        p = prev[col].iloc[0]
        if p is None or p == 0 or pd.isna(p):
            return {"value": val, "delta": np.nan}

        return {"value": val, "delta": (val - p) / p}

    mapping = {
        "Spend": extract("Spend"),
        "Traffic": extract("Traffic"),
        "Engaged_Visits": extract("Engaged Visits") if "Engaged Visits" in df.columns else extract("Engaged_Visits"),
        "EVR": extract("EVR"),
    }

    return mapping


def campaign_summary_text(df_campaign):
    """
    Adobe-style readable summary text block for a campaign.
    Includes:
    - contribution %
    - EVR movement
    - platform contribution (if available)
    """
    if df_campaign.empty:
        return "No campaign data available."

    df = df_campaign.sort_values(["Year", "Quarter", "Week"]).copy()

    latest = df.tail(1)
    prev = df.tail(2).head(1) if len(df) >= 2 else None

    # EVR derive if missing
    if "EVR" not in df.columns:
        if "Traffic" in df.columns and "Engaged Visits" in df.columns:
            df["EVR"] = np.where(df["Traffic"] > 0,
                                 df["Engaged Visits"] / df["Traffic"],
                                 np.nan)
        latest = df.tail(1)
        prev = df.tail(2).head(1) if len(df) >= 2 else None

    now_evr = latest["EVR"].iloc[0]
    prev_evr = prev["EVR"].iloc[0] if prev is not None and "EVR" in prev.columns else np.nan

    if pd.isna(prev_evr):
        evr_line = f"Engagement rate sits at {now_evr*100:.1f}%."
    else:
        movement = "up" if now_evr > prev_evr else "down"
        evr_line = f"Engagement rate is {now_evr*100:.1f}% and moved {movement} WoW."

    # Contribution %
    if "Engaged Visits" in df.columns:
        total = df["Engaged Visits"].sum()
        contrib = latest["Engaged Visits"].iloc[0] / total * 100 if total > 0 else np.nan
        contrib_line = f"Contributes {contrib:.1f}% of this campaign’s engaged visits this week."
    else:
        contrib_line = ""

    # Platform highest contribution
    if "Platform" in df.columns and "Engaged Visits" in df.columns:
        grouped = df.groupby("Platform")["Engaged Visits"].sum()
        top = grouped.idxmax()
        platform_line = f"Highest contribution comes from **{top}**."
    else:
        platform_line = ""

    return "\n\n".join([contrib_line, evr_line, platform_line]).strip()
