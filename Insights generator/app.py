from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np


from src.config import FilterConfig, MIN_REQUIRED
from src.data_loader import load_data, normalise_schema
from src.filters import apply_filters
from src.features import build_all_aggregates
from src.ml_models import detect_anomalies

from src.visuals import (
    line_metric_over_time,
    bar_qtd_by_dimension,
    table_from_dataframe,
    stacked_share_bar,
    efficiency_scatter,
    heatmap_by_dimension_week,
    campaign_channel_metrics_html,
    campaign_channel_insights_html,
    campaign_channel_insights_text_blocks,
    latest_week_dimension_bar,
    latest_week_dimension_pie,
)

from src.reporting import (
    build_headline_insights,
    build_channel_insights,
    build_market_insights,
    build_seasonality_insights,
    build_mix_efficiency_insights,
    build_volatility_insights,
    build_driver_insights,
)


def _safe_default_data_path() -> Optional[Path]:
    default = Path("data") / "data.xlsx"
    return default if default.exists() else None


@st.cache_data(show_spinner=False)
def _load_from_streamlit_upload(uploaded_file) -> pd.DataFrame:
    """
    Reads the uploaded file and normalises schema
    so the app supports multiple dataset formats.
    """
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = normalise_schema(df)
    return df


def main():
    st.set_page_config(
        page_title="Automated media insights",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Global styling
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            padding-left: 4rem;
            padding-right: 4rem;
        }
        .stApp {
            background-color: #f3f4f6;
        }
        h1, h2, h3 {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .insight-card {
            background: #ffffff;
            padding: 1.25rem 1.5rem;
            border-radius: 14px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
        }
        .slide-card {
            background: #ffffff;
            padding: 1.25rem 1.5rem;
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
            border: 1px solid #eef2f7;
        }
        .pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            background: #f1f5f9;
            color: #0f172a;
            font-size: 11px;
            font-weight: 650;
            margin-right: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Automated media insights")

    # Sidebar, data + filters
    with st.sidebar:
        st.header("Data input")

        uploaded = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
        )

        # -------------------------
        # Load data
        # -------------------------
        try:
            if uploaded is not None:
                df_raw = _load_from_streamlit_upload(uploaded)
            else:
                default_path = _safe_default_data_path()
                if default_path is None:
                    st.info(
                        "Upload a CSV or Excel file to begin, "
                        "or place `data.xlsx` inside a `data/` folder in the project root."
                    )
                    st.stop()

                df_raw = load_data(default_path)

        except Exception as e:
            st.error(f"Upload failed during normalisation: {e}")
            st.stop()

        # -------------------------
        # Minimal validation
        # -------------------------
        missing = [c for c in MIN_REQUIRED if c not in df_raw.columns]
        if missing:
            st.error(
                f"Dataset missing minimum required columns after normalisation: {missing}"
            )
            st.stop()

        for c in ["Week", "Quarter", "Year"]:
            if c in df_raw.columns:
                df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").astype("Int64")

        st.success("Data loaded successfully.")

        # -------------------------
        # Filters
        # -------------------------
        st.header("Filters")

        years = (
            sorted(df_raw["Year"].dropna().unique().tolist())
            if "Year" in df_raw.columns
            else []
        )
        selected_years = st.multiselect(
            "Years",
            options=years,
            default=years[-1:] if years else [],
        )

        df_for_period = df_raw.copy()
        if selected_years and "Year" in df_for_period.columns:
            df_for_period = df_for_period[df_for_period["Year"].isin(selected_years)]

        quarters = (
            sorted(df_for_period["Quarter"].dropna().unique().tolist())
            if "Quarter" in df_for_period.columns
            else []
        )

        quarter = st.selectbox(
            "Quarter",
            options=["All"] + quarters,
            index=(quarters.index(max(quarters)) + 1) if quarters else 0,
        )

        if quarter != "All" and "Quarter" in df_for_period.columns:
            df_for_weeks = df_for_period[df_for_period["Quarter"] == quarter]
        else:
            df_for_weeks = df_for_period

        weeks = (
            sorted(df_for_weeks["Week"].dropna().unique().tolist())
            if "Week" in df_for_weeks.columns
            else []
        )

        if weeks:
            min_week, max_week = int(min(weeks)), int(max(weeks))
        else:
            min_week, max_week = 1, 52

        if min_week >= max_week:
            st.caption(f"Only one week available in this selection: W{min_week}")
            start_week, end_week = min_week, max_week
        else:
            start_week, end_week = st.slider(
                "Week range",
                min_value=min_week,
                max_value=max_week,
                value=(min_week, max_week),
                step=1,
            )

        def _multi(label: str, col: str):
            if col not in df_for_period.columns:
                return []
            options = sorted(df_for_period[col].dropna().astype(str).unique().tolist())
            return st.multiselect(label, options=options, default=[])

        platforms = _multi("Platforms", "Platform")
        campaigns = _multi("Campaigns", "Campaign")
        markets = _multi("Markets", "Market")
        channels = _multi("Channels", "Channel")
        devices = _multi("Devices", "Device")

        run_button = st.button("Run analysis", type="primary")

    # -------------------------
    # Run
    # -------------------------
    if not run_button:
        st.info("Adjust filters in the sidebar then click **Run analysis**.")
        st.stop()

    filt_cfg = FilterConfig(
        quarter=None if quarter == "All" else int(quarter),
        start_week=int(start_week),
        end_week=int(end_week),
        years=[int(y) for y in selected_years] if selected_years else [],
        campaigns=campaigns,
        markets=markets,
        channels=channels,
        devices=devices,
        platforms=platforms,
    )

    df_filtered = apply_filters(df_raw, filt_cfg)

    if df_filtered.empty:
        st.warning("No data matched the selected filters. Try relaxing the filters.")
        st.stop()

    # Aggregations and modelling
    agg = build_all_aggregates(df_filtered)

    overall_weekly = agg["overall_weekly"]
    overall_weekly = detect_anomalies(
        overall_weekly,
        group_cols=[],
        metric_cols=["Spend", "Traffic", "Engaged_Visits"],
        contamination=0.15,
    )
    agg["overall_weekly"] = overall_weekly

    channel_qtd = agg["channel_qtd"]
    market_qtd = agg["market_qtd"]
    device_qtd = agg["device_qtd"]
    channel_weekly = agg["channel_weekly"]
    market_weekly = agg["market_weekly"]

    # Narrative building (existing)
    headline_insights = build_headline_insights(overall_weekly)
    channel_insights = build_channel_insights(channel_qtd)
    market_insights = build_market_insights(market_qtd)
    seasonality_insights = build_seasonality_insights(overall_weekly)
    mix_insights = build_mix_efficiency_insights(channel_qtd)
    volatility_insights = build_volatility_insights(channel_weekly, market_weekly)
    driver_insights = build_driver_insights(df_filtered)

    # -----------------------------------------------------------------
    # NEW: Stakeholder-friendly Campaign x Channel metrics table
    # -----------------------------------------------------------------
    st.subheader("Campaign x Channel performance (latest week)")

    ccm_html = campaign_channel_metrics_html(
        df_filtered,
        metrics=[
            {"label": "Spend Pacing", "key": "Spend", "kind": "money"},
            {"label": "Visit Pacing", "key": "Traffic", "kind": "number"},
            {"label": "Engaged Visits", "key": "Engaged Visits", "kind": "number"},
            {"label": "EVR", "key": "EVR", "kind": "pct"},
            {"label": "CPV", "key": "CPV", "kind": "money"},
            {"label": "CPeV", "key": "CPE", "kind": "money"},
        ],
        include_insights=True,
    )

    st.markdown(ccm_html, unsafe_allow_html=True)
    st.caption("Grouped view by campaign and channel, showing latest week values with WoW change and light-touch commentary.")


    # ---------------------------------------------------------------------
    # NEW: Latest week performance comparisons
    # ---------------------------------------------------------------------
    st.subheader("Latest week performance comparisons")

    # Nice two-row layout
    row1 = st.columns(2)
    row2 = st.columns(2)
    row3 = st.columns(2)

    # 1) Visits pacing by channel
    fig = latest_week_dimension_bar(
        df_filtered,
        dimension="Channel",
        metric_key="Traffic",
        title="Visit pacing by channel (latest week)",
        top_n=12,
    )
    if fig.data:
        row1[0].plotly_chart(fig, use_container_width=True)

    # Optional share view
    fig = latest_week_dimension_pie(
        df_filtered,
        dimension="Channel",
        metric_key="Traffic",
        title="Visit share by channel (latest week)",
        top_n=6,
    )
    if fig.data:
        row1[1].plotly_chart(fig, use_container_width=True)

    # 2) Spend by channel
    fig = latest_week_dimension_bar(
        df_filtered,
        dimension="Channel",
        metric_key="Spend",
        title="Spend pacing by channel (latest week)",
        top_n=12,
    )
    if fig.data:
        row2[0].plotly_chart(fig, use_container_width=True)

    fig = latest_week_dimension_pie(
        df_filtered,
        dimension="Channel",
        metric_key="Spend",
        title="Spend share by channel (latest week)",
        top_n=6,
    )
    if fig.data:
        row2[1].plotly_chart(fig, use_container_width=True)

    # 3) Engaged visits by channel
    fig = latest_week_dimension_bar(
        df_filtered,
        dimension="Channel",
        metric_key="Engaged Visits",
        title="Engaged visits by channel (latest week)",
        top_n=12,
    )
    if fig.data:
        row3[0].plotly_chart(fig, use_container_width=True)

    # 4) EVR by Device / Platform
    # Put these under the last right-hand slot if you prefer a clean 2-column stack
    with row3[1]:
        fig = latest_week_dimension_bar(
            df_filtered,
            dimension="Device",
            metric_key="EVR",
            title="EVR by device (latest week)",
            top_n=10,
        )
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)

        if "Platform" in df_filtered.columns:
            fig = latest_week_dimension_bar(
                df_filtered,
                dimension="Platform",
                metric_key="EVR",
                title="EVR by platform (latest week)",
                top_n=10,
            )
            if fig.data:
                st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Latest-week comparison views to support quick, visual storytelling across investment, demand, and quality."
    )

    # -----------------------------------------------------------------
    # NEW: Copy-ready Campaign x Channel insights (HTML)
    # -----------------------------------------------------------------
    st.subheader("Campaign x Channel copy-ready insights")

    cci_html = campaign_channel_insights_html(
        df_filtered,
        max_bullets=3,
    )
    st.markdown(cci_html, unsafe_allow_html=True)
    st.caption("Auto-generated insights for the latest week in your selection.")

    # -----------------------------------------------------------------
    # NEW: Copy snippets (text areas)
    # -----------------------------------------------------------------
    st.subheader("Copy snippets for slides")

    blocks = campaign_channel_insights_text_blocks(
        df_filtered,
        max_bullets=3,
    )

    if not blocks:
        st.info("No copy snippets available for this selection.")
    else:
        for camp, text in blocks.items():
            with st.expander(f"Copy block — {camp}", expanded=True):
                st.text_area(
                    label="",
                    value=text,
                    height=140,
                    key=f"copy_block_{camp}",
                )

    # -----------------------------------------------------------------
    # NEW: Slide-ready one-pager preview
    # -----------------------------------------------------------------
    st.subheader("Slide-ready one-pager preview")

    # Build quick headline numbers from latest overall week if available
    latest_overall = None
    if overall_weekly is not None and not overall_weekly.empty:
        latest_overall = (
            overall_weekly.sort_values(["Year", "Quarter", "Week"])
            .tail(1)
            .iloc[0]
        )

    # Create a compact view that mirrors your PPT vibe
    st.markdown("<div class='slide-card'>", unsafe_allow_html=True)

    if latest_overall is not None:
        y = int(latest_overall.get("Year", 0)) if pd.notna(latest_overall.get("Year", np.nan)) else None
        q = int(latest_overall.get("Quarter", 0)) if pd.notna(latest_overall.get("Quarter", np.nan)) else None
        w = int(latest_overall.get("Week", 0)) if pd.notna(latest_overall.get("Week", np.nan)) else None

        label = "Latest week"
        if y and q and w:
            label = f"Latest week snapshot — {y} Q{q} W{w}"

        st.markdown(f"**{label}**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Spend", f"£{float(latest_overall.get('Spend', 0) or 0):,.0f}")
        with col2:
            st.metric("Traffic", f"{float(latest_overall.get('Traffic', 0) or 0):,.0f}")
        with col3:
            ev = float(latest_overall.get("Engaged_Visits", 0) or 0)
            st.metric("Engaged Visits", f"{ev:,.0f}")
        with col4:
            evr = latest_overall.get("EVR", np.nan)
            evr_str = f"{float(evr)*100:.1f}%" if pd.notna(evr) else "—"
            st.metric("EVR", evr_str)

        st.markdown("---")

    # Campaign blocks
    if blocks:
        for camp, text in blocks.items():
            st.markdown(f"<span class='pill'>Campaign</span> **{camp}**", unsafe_allow_html=True)
            # Show bullet-style
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            # skip the first header line because we already print camp name
            for ln in lines[1:]:
                st.markdown(f"- {ln.replace('• ', '')}")
            st.markdown("")

    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("This section is designed to mirror a one-page stakeholder slide. Copy bullets directly.")

    # -----------------------------------------------------------------
    # Existing broader narrative section
    # -----------------------------------------------------------------
    st.subheader("Key insights")

    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown("**Headline performance**")
        st.caption("Summarises the latest week in the selected period, including week on week changes for spend, traffic and EVR.")
        for line in headline_insights:
            st.markdown(f"- {line}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown("**Seasonality and trends**")
        st.caption("Looks at how spend, traffic and engagement evolve from the start to the end of the selected period.")
        for line in seasonality_insights:
            st.markdown(f"- {line}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown("**Channel performance**")
        st.caption("Shows how engaged visits and spend are distributed across channels.")
        for line in channel_insights:
            st.markdown(f"- {line}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown("**Market performance**")
        st.caption("Highlights the markets that drive the largest engaged visit volumes.")
        for line in market_insights:
            st.markdown(f"- {line}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown("**Mix and efficiency opportunities**")
        st.caption("Identifies channels whose engagement share is higher or lower than their share of spend.")
        for line in mix_insights:
            st.markdown(f"- {line}")
        st.markdown("</div>", unsafe_allow_html=True)

        if volatility_insights:
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown("**Volatility and stability**")
            st.caption("Highlights channels or markets where weekly results swing more than average.")
            for line in volatility_insights:
                st.markdown(f"- {line}")
            st.markdown("</div>", unsafe_allow_html=True)

        if driver_insights:
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown("**Drivers and modelling**")
            st.caption("A Random Forest model estimates which variables are most strongly associated with engaged visits.")
            for line in driver_insights:
                st.markdown(f"- {line}")
            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # Time series
    # -----------------------------------------------------------------
    st.subheader("Time series overview")

    if overall_weekly is not None and not overall_weekly.empty:
        st.plotly_chart(
            line_metric_over_time(overall_weekly, "Spend", "Spend by week"),
            use_container_width=True,
        )
        st.plotly_chart(
            line_metric_over_time(overall_weekly, "Traffic", "Traffic by week"),
            use_container_width=True,
        )
        st.plotly_chart(
            line_metric_over_time(overall_weekly, "EVR", "EVR by week"),
            use_container_width=True,
        )

    # -----------------------------------------------------------------
    # Channel and market
    # -----------------------------------------------------------------
    st.subheader("Channel and market performance")

    if channel_qtd is not None and not channel_qtd.empty:
        st.plotly_chart(
            bar_qtd_by_dimension(
                channel_qtd,
                "Channel",
                "Engaged_Visits_QTD",
                "Engaged visits by channel (QTD)",
            ),
            use_container_width=True,
        )

        ch_share = channel_qtd.copy()
        total_spend = float(ch_share["Spend_QTD"].sum()) if "Spend_QTD" in ch_share.columns else 0.0
        total_eng = float(ch_share["Engaged_Visits_QTD"].sum()) if "Engaged_Visits_QTD" in ch_share.columns else 0.0
        if total_spend == 0:
            total_spend = 1.0
        if total_eng == 0:
            total_eng = 1.0

        if "Spend_QTD" in ch_share.columns:
            ch_share["Spend_share"] = ch_share["Spend_QTD"] / total_spend
        if "Engaged_Visits_QTD" in ch_share.columns:
            ch_share["Eng_share"] = ch_share["Engaged_Visits_QTD"] / total_eng

        st.plotly_chart(
            stacked_share_bar(
                ch_share,
                "Channel",
                {"Spend_share": "Spend share", "Eng_share": "Engaged visits share"},
                "Channel spend vs engagement share (QTD)",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            efficiency_scatter(
                channel_qtd,
                "Channel",
                "Channel mix efficiency (spend vs engagement share)",
            ),
            use_container_width=True,
        )

        cols = [c for c in ["Channel", "Spend_QTD", "Traffic_QTD", "Engaged_Visits_QTD", "EVR_QTD", "CPE_QTD"] if c in channel_qtd.columns]
        st.plotly_chart(
            table_from_dataframe(channel_qtd[cols], "Channel QTD summary"),
            use_container_width=True,
        )

    if market_qtd is not None and not market_qtd.empty:
        st.plotly_chart(
            bar_qtd_by_dimension(
                market_qtd,
                "Market",
                "Engaged_Visits_QTD",
                "Engaged visits by market (QTD)",
            ),
            use_container_width=True,
        )

        cols = [c for c in ["Market", "Spend_QTD", "Traffic_QTD", "Engaged_Visits_QTD", "EVR_QTD", "CPE_QTD"] if c in market_qtd.columns]
        st.plotly_chart(
            table_from_dataframe(market_qtd[cols], "Market QTD summary"),
            use_container_width=True,
        )

    # -----------------------------------------------------------------
    # Heatmap
    # -----------------------------------------------------------------
    st.subheader("Channel EVR heatmap")

    if channel_weekly is not None and not channel_weekly.empty:
        st.plotly_chart(
            heatmap_by_dimension_week(
                channel_weekly,
                "Channel",
                "EVR",
                "EVR by channel and week",
            ),
            use_container_width=True,
        )

    # -----------------------------------------------------------------
    # Device and platform
    # -----------------------------------------------------------------
    st.subheader("Device and platform mix")

    if device_qtd is not None and not device_qtd.empty:
        cols = [c for c in ["Device", "Spend_QTD", "Traffic_QTD", "Engaged_Visits_QTD", "EVR_QTD", "CPE_QTD"] if c in device_qtd.columns]
        st.plotly_chart(
            table_from_dataframe(device_qtd[cols], "Device QTD summary"),
            use_container_width=True,
        )

    if "Platform" in df_filtered.columns:
        plat = (
            df_filtered.groupby("Platform", dropna=False)
            .agg(
                Spend=("Spend", "sum"),
                Traffic=("Traffic", "sum"),
                Engaged_Visits=("Engaged Visits", "sum"),
            )
            .reset_index()
        )
        if not plat.empty:
            plat["CPE"] = plat["Spend"] / plat["Engaged_Visits"].where(plat["Engaged_Visits"] > 0)

            cols = [c for c in ["Platform", "Spend", "Traffic", "Engaged_Visits", "CPE"] if c in plat.columns]
            st.plotly_chart(
                table_from_dataframe(plat[cols], "Platform performance summary"),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
