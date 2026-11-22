"""Builds work/events.csv and work/outages.csv from the ten 24-hour-lag files in data/."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # repo root, so it runs from any folder

(ROOT / "work").mkdir(exist_ok=True)
files = sorted((ROOT / "data").glob("eaglei_outages_with_events_*_24_hours_lag.csv"))
df = pd.concat(pd.read_csv(f).assign(Year=int(f.name.split("_")[4])) for f in files)
df["EventKey"] = df["Year"].astype(str) + "-" + df["event_id"]
df = df.drop_duplicates(subset=["EventKey", "fips", "start_time"])
df["CustomerHours"] = df["mean_customers"] * df["duration"]

events = df.groupby("EventKey").agg(
    Year=("Year", "first"), State=("state", "first"), EventType=("Event Type", "first"),
    Began=("Datetime Event Began", "first"), CountyOutages=("fips", "size"),
    CustomerHours=("CustomerHours", "sum")).reset_index()
weather = events["EventType"].str.lower().str.contains("weather|natural disaster|storm")
events["Weather"] = weather.map({True: "Yes", False: "No"})
events["Month"] = pd.to_datetime(events["Began"]).dt.month
events["HighImpact"] = events["CustomerHours"] >= 500_000

outages = df.drop_duplicates(subset=["fips", "start_time"])   # each county outage once
outages = outages[["fips", "state", "county", "start_time", "duration", "mean_customers", "CustomerHours"]]

events.to_csv(ROOT / "work" / "events.csv", index=False)
outages.to_csv(ROOT / "work" / "outages.csv", index=False)
print(f"{len(events):,} events, {len(outages):,} distinct county outages")   # expect 2,683 and 79,711
