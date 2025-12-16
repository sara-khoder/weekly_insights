import pandas as pd
from src.campaign_ppt_builder import CampaignPPTBuilder

df = pd.read_excel("data.xlsx")  # or CSV
builder = CampaignPPTBuilder(
    template_path="Q1W1.pptx",
    output_path="Campaign_Report.pptx"
)
builder.build(df)
