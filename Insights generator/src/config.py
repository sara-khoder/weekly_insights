from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ---------------------------------------------------------------------
# Canonical schema support (NEW)
# ---------------------------------------------------------------------
# These aliases let your loader map different input datasets
# into a stable internal schema.
CANONICAL_ALIASES: Dict[str, List[str]] = {
    # Dimensions
    "Campaign": ["Campaign", "Belongs_to_this_Campaign"],
    "Platform": ["Platform"],
    "Market": ["Market", "Market_Area", "Geo", "Region", "Country"],
    "Channel": ["Channel", "Marketing_Channel_Type", "Marketing Channel Type"],
    "Device": ["Device"],

    # Time
    "Date": ["Date"],
    "Week": ["Week", "ISO_Week"],
    "Quarter": ["Quarter"],
    "Year": ["Year"],
    "YearWeek": ["YearWeek", "Year_Week"],

    # Core metrics (weekly/daily naming differences)
    "Spend": ["Spend", "Actual_Weekly_Spend", "Spend_Weekly_Actual", "Spend_Daily_Actual"],
    "Traffic": ["Traffic", "Traffic_Weekly_Actual", "Traffic_Daily_Actual", "Visits"],
    "Clicks": ["Clicks", "Clicks_Weekly_Actual", "Clicks_Daily_Actual"],
    "Impressions": ["Impressions", "Impressions_Weekly_Actual", "Impressions_Daily_Actual"],
    "Engaged Visits": ["Engaged Visits", "~Engaged Visits", "Engaged_Visits"],

    # Optional / social / video style metrics
    "Video Start": ["Video Start", "Video Starts"],
    "Video End": ["Video End", "Video Ends"],
    "Video Completion Rate": ["Video Completion Rate"],
    "Likes/Reactions": ["Likes/Reactions", "Likes Reactions", "Reactions", "Likes"],
    "Shares": ["Shares"],
    "Comments": ["Comments"],
    "Non-GTM Total Seconds Spent": ["Non-GTM Total Seconds Spent"],
}

# Minimal requirements AFTER normalisation.
# This is what your new data_loader should validate.
MIN_REQUIRED = [
    "Campaign",
    "Platform",
    # time is derived from Date where possible
    "Spend",
    "Traffic",
]

TIME_COLUMNS = ["Year", "Quarter", "Week"]


# ---------------------------------------------------------------------
# Original project constants (KEPT for backwards compatibility)
# ---------------------------------------------------------------------
# If any older part of the code still references REQUIRED_COLUMNS,
# this list remains as-is.
REQUIRED_COLUMNS = [
    "Campaign",
    "Platform",
    "Market",
    "Week",
    "Quarter",
    "Year",
    "Channel",
    "Device",
    "Spend",
    "Traffic",
    "Clicks",
    "Impressions",
    "CPC",
    "CPV",
    "CPM",
    "CTR",
    "Engaged Visits",
    "Engaged Visits Rate",
    "Visit Rate",
    "Bounce Rate",
    "Cost Per Engaged Visits",
]

# Numeric columns used across modelling/cleaning
NUMERIC_COLUMNS = [
    "Spend",
    "Traffic",
    "Clicks",
    "Impressions",
    "CPC",
    "CPV",
    "CPM",
    "CTR",
    "Engaged Visits",
    "Engaged Visits Rate",
    "Visit Rate",
    "Bounce Rate",
    "Cost Per Engaged Visits",
    "Video Start",
    "Video End",
    "Video Completion Rate",
    "Likes/Reactions",
    "Shares",
    "Comments",
    "Non-GTM Total Seconds Spent",
]

# Dimensions used in modelling
DIMENSION_COLUMNS = [
    "Campaign",
    "Platform",
    "Market",
    "Channel",
    "Device",
]


# ---------------------------------------------------------------------
# Insight thresholds + filter config (original behaviour)
# ---------------------------------------------------------------------
@dataclass
class InsightThresholds:
    # Channel and market mix
    share_gap_strong: float = 0.03  # share of EV vs share of spend gap

    # Correlation for seasonality
    corr_min_abs: float = 0.3

    # Market efficiency bands
    evr_hot_uplift: float = 0.10    # 10 percent above average
    cpe_hot_saving: float = 0.05    # 5 percent cheaper than average
    evr_cold_drop: float = 0.10     # 10 percent below average
    cpe_cold_penalty: float = 0.10  # 10 percent more expensive

    # Headline comment threshold
    min_rel_change_to_comment: float = 0.05  # 5 percent move before we talk about it


INSIGHT_THRESHOLDS = InsightThresholds()


@dataclass
class FilterConfig:
    quarter: Optional[int] = None
    start_week: Optional[int] = None
    end_week: Optional[int] = None
    years: List[int] = field(default_factory=list)
    campaigns: List[str] = field(default_factory=list)
    markets: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    devices: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)


DEFAULT_PRIMARY_METRIC = "Engaged Visits"
DEFAULT_SECONDARY_METRIC = "Cost Per Engaged Visits"
