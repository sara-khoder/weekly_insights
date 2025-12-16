
import argparse
from pathlib import Path

from .data_loader import load_data
from .filters import parse_filter_args, apply_filters
from .features import build_all_aggregates
from .ml_models import detect_anomalies
from .visuals import (
    line_metric_over_time,
    bar_qtd_by_dimension,
    table_from_dataframe,
)

from .reporting import (
    build_headline_insights,
    build_anomaly_insights,
    build_channel_insights,
    build_market_insights,
    build_mix_efficiency_insights,
    build_volatility_insights,
    build_driver_insights,
    build_contributor_insights, 
    build_html_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automated media insights generator",
    )
    p.add_argument("--data-path", required=True, help="Path to CSV or XLSX data file")
    p.add_argument("--output-dir", default="reports", help="Directory for HTML reports")
    p.add_argument("--output-name", default="insights_qtd.html", help="Output HTML file name")

    p.add_argument("--years", type=str, default="", help="Comma separated list of years")
    p.add_argument("--quarter", type=int, default=None, help="Quarter to analyse, for example 4")
    p.add_argument("--start-week", type=int, default=None, help="Minimum ISO week number")
    p.add_argument("--end-week", type=int, default=None, help="Maximum ISO week number")

    p.add_argument("--campaigns", type=str, default="", help="Comma separated list of campaigns")
    p.add_argument("--markets", type=str, default="", help="Comma separated list of markets")
    p.add_argument("--channels", type=str, default="", help="Comma separated list of channels")
    p.add_argument("--devices", type=str, default="", help="Comma separated list of devices")
    p.add_argument("--platforms", type=str, default="", help="Comma separated list of platforms")

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    df = load_data(args.data_path)
    filt_cfg = parse_filter_args(args)
    df_filtered = apply_filters(df, filt_cfg)

    agg = build_all_aggregates(df_filtered)

    overall_weekly = agg["overall_weekly"]
    channel_weekly = agg["channel_weekly"]
    market_weekly = agg["market_weekly"]

    overall_weekly = detect_anomalies(
        overall_weekly,
        group_cols=[],
        metric_cols=["Spend", "Traffic", "Engaged_Visits"],
        contamination=0.15,
    )
    channel_weekly = detect_anomalies(
        channel_weekly,
        group_cols=["Channel"],
        metric_cols=["Spend", "Traffic", "Engaged_Visits"],
        contamination=0.15,
    )
    market_weekly = detect_anomalies(
        market_weekly,
        group_cols=["Market"],
        metric_cols=["Spend", "Traffic", "Engaged_Visits"],
        contamination=0.15,
    )

    agg["overall_weekly"] = overall_weekly
    agg["channel_weekly"] = channel_weekly
    agg["market_weekly"] = market_weekly

    figures = {}

    if not overall_weekly.empty:
        figures["QTD spend by week"] = line_metric_over_time(
            overall_weekly, metric="Spend", title="Spend by week"
        )
        figures["QTD traffic by week"] = line_metric_over_time(
            overall_weekly, metric="Traffic", title="Traffic by week"
        )
        figures["QTD EVR by week"] = line_metric_over_time(
            overall_weekly, metric="EVR", title="EVR by week"
        )

    channel_qtd = agg["channel_qtd"]
    if not channel_qtd.empty:
        figures["QTD engaged visits by channel"] = bar_qtd_by_dimension(
            channel_qtd,
            dimension="Channel",
            value_col="Engaged_Visits_QTD",
            title="Engaged visits by channel (QTD)",
        )

    market_qtd = agg["market_qtd"]
    if not market_qtd.empty:
        figures["QTD engaged visits by market"] = bar_qtd_by_dimension(
            market_qtd,
            dimension="Market",
            value_col="Engaged_Visits_QTD",
            title="Engaged visits by market (QTD)",
        )

    figures["Channel QTD table"] = table_from_dataframe(
        channel_qtd[["Channel", "Spend_QTD", "Traffic_QTD", "Engaged_Visits_QTD", "EVR_QTD", "CPE_QTD"]],
        title="Channel QTD summary",
    )
    figures["Market QTD table"] = table_from_dataframe(
        market_qtd[["Market", "Spend_QTD", "Traffic_QTD", "Engaged_Visits_QTD", "EVR_QTD", "CPE_QTD"]],
        title="Market QTD summary",
    )



    text_blocks = {
        "headline": build_headline_insights(overall_weekly),
        "anomalies": build_anomaly_insights(overall_weekly),
        "seasonality": build_seasonality_insights(overall_weekly),
        "channel": build_channel_insights(channel_qtd),
        "contributors_channel": build_contributor_insights(agg["channel_weekly"], dim="Channel"),
        "market": build_market_insights(market_qtd),
        "contributors_market": build_contributor_insights(agg["market_weekly"], dim="Market"),
        "mix": build_mix_efficiency_insights(channel_qtd),
        "volatility": build_volatility_insights(agg["channel_weekly"], agg["market_weekly"]),
        "drivers": build_driver_insights(df_filtered),
    }

    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name
    build_html_report(output_path, figures, text_blocks)

    print(f"Report written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
