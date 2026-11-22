# grid-agent
Two agents over U.S. outage data (DOE-417 events, Nov 2014 to 2023).
- work/events.csv has one row per event; work/outages.csv counts each county outage once.
  Totals and state rankings come from outages.csv.
- Never touch the 2022-23 test years except through `uv run python src/coach.py final`.
- Before saying a number is right, recompute it with pandas and show the code.
- Secrets live in .env and are never committed.