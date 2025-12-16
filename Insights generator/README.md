
# Automated Media Insights ML Project

This project generates automated, machine learning informed insights from paid media data.

## Key features

- Year, quarter and week range selection
- Filters for Campaign, Market, Channel, Device, Platform
- Automated weekly and QTD aggregates
- IsolationForest based anomaly detection for unusual weeks
- RandomForest based feature importance for engagement drivers
- Interactive Streamlit dashboard with modern visuals
- Optional HTML report generation from the CLI

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place your dataset in `data/data.xlsx` (or upload it via the Streamlit UI). The dataset should contain at least:

- Campaign, Platform, Market, Week, Quarter, Year, Channel, Device
- Spend, Traffic, Clicks, Impressions
- Engaged Visits, Engaged Visits Rate, Cost Per Engaged Visits

3. Run the dashboard:

```bash
streamlit run app.py
```

4. Adjust filters in the sidebar and click **Run analysis** to generate insights, charts and tables.
