# Grid Reliability: which major outage events become high-impact?

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-star%20schema%20%2B%20DAX-F2C811?logo=powerbi&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-logistic%20regression-F7931E?logo=scikitlearn&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-2%20agents-D97757?logo=anthropic&logoColor=white)
![Tests](https://img.shields.io/badge/evals-5%2F5-2E7D32)

A Power BI report, an ML model and two Claude agents built on ten years of U.S. power outage data
(Nov 2014 to 2023): 2,683 major disturbance events (DOE-417) linked to 79,711 county-level outages
(EAGLE-I).

> **The question:** when a major event starts, can we tell early which ones will grow into
> **high-impact** events (≥ 500,000 customer-hours without power)?

## At a glance

| | |
|---|---|
| **2.55 billion** customer-hours lost | across 79,711 distinct county outages tied to 2,683 major events |
| **6% of outages, 33% of the impact** | major events vs. all 1.39 million recorded outages |
| **24% of events carry 95% of the hours** | the 649 high-impact events dominate everything else |
| **Weather: 49% of events, 89% of high-impact ones** | and 96% of all major-event customer-hours |
| **Recall 0.73 on unseen years** | the model catches 86 of 118 high-impact events in 2022–2023 |
| **5 / 5 evals, about $0.014 a question** | the analyst agent, scored on questions with known answers |

## Contents
- [How it fits together](#how-it-fits-together)
- [Data model: the star schema](#data-model-the-star-schema)
- [The Power BI report](#the-power-bi-report)
- [The model](#the-model)
- [The agents](#the-agents)
- [Limitations](#limitations)
- [Run it](#run-it)

## How it fits together

```
 data/   ten yearly CSVs: EAGLE-I county outages matched to DOE-417 events (24-hour window)
   │
   ├──► Power Query ──► star schema ──► DAX measures ──► Power BI report (3 pages)
   │                                                            ▲
   ├──► src/outage_model.py ──► outputs/event_risk.csv ─────────┘  risk score per event
   │
   └──► src/make_events.py ──► work/events.csv    one row per event
                               work/outages.csv   each county outage once
                                    │
                                    ├──► src/agent.py   analyst agent: question → pandas → answer
                                    ├──► src/coach.py   model coach: tunes the model, test years locked
                                    └──► src/evals.py   re-runs the analyst on known answers
```

The rule throughout: **known steps go in plain code, and the agents get the questions you can't
predict.** Cleaning, de-duplication and the model are ordinary scripts. The agents sit on top of the
clean tables, not the raw files.

## Data model: the star schema

Two fact tables share two dimensions, so one Year or State slicer filters every visual on the page.

```
                                ┌────────────────────────────┐       ┌───────────────────────┐
                                │ DimEvent        [dim]      │       │ event_risk            │
                                │ one row per event          │       │ Python model output   │
                                ├────────────────────────────┤ 1   1 ├───────────────────────┤
                                │ EventKey            PK     ├───────┤ EventKey          PK  │
                                │ State                      │       │ Risk         0 to 1   │
                                │ EventBegan                 │       └───────────────────────┘
                                │ EventType                  │
                                │ CauseCategory    (7)       │
                                │ TotalCustomerHours         │
                                │ Season                     │
                                │ ImpactLevel  ◄ ML target   │
                                └──────────────┬─────────────┘
                                               │ 1
                                               │ EventKey
                                               │ *
                                ┌──────────────┴─────────────┐
                                │ EventOutages   [fact]      │
                                │ one row per event × outage │
                                ├────────────────────────────┤
                                │ EventKey            FK     │
              OutageDate      * │ OutageDate          FK     │ *       State
            ┌───────────────────┤ State               FK     ├───────────────────┐
            │                   │ OutageID                   │                   │
            │                   │ FIPS                       │                   │
            │                   │ EventBegan                 │                   │
            │                   │ EventType                  │                   │
            │                   │ CauseCategory              │                   │
            │                   │ DurationHrs                │                   │
            │                   │ CustomerHours              │                   │
            │                   └────────────────────────────┘                   │
            │                                                                    │
            │ 1                                                                  │ 1
┌───────────┴───────────┐                                            ┌───────────┴───────────┐
│ DimDate         [dim] │                                            │ DimState        [dim] │
│ one row per day       │                                            │ one row per state     │
├───────────────────────┤                                            ├───────────────────────┤
│ Date              PK  │                                            │ State             PK  │
│ Year                  │                                            └───────────┬───────────┘
│ Month                 │                                                        │ 1
│ MonthStart            │                                                        │
└───────────┬───────────┘                                                        │
            │ 1                                                                  │
            │                   ┌────────────────────────────┐                   │
            │                   │ MonthlySummary [fact]      │                   │
            │                   │ one row per state × month  │                   │
            │                   ├────────────────────────────┤                   │
            │ MonthStart      * │ MonthStart          FK     │ *       State     │
            └───────────────────┤ State               FK     ├───────────────────┘
                                │ outage_count               │
                                │ max_outage_duration        │
                                │ customer_weighted_hours    │
                                └────────────────────────────┘
```

- **Five relationships, all many-to-one with single-direction filtering** (dimension → fact).
  Setting any of them to *Both* created two filter paths between the same tables and an
  "ambiguous paths" error.
- **Two facts at different grains.** `EventOutages` holds the major-event outages; `MonthlySummary`
  holds monthly totals for *all* outages. Sharing `DimDate` and `DimState` lets the report put them
  side by side, which is how the 6% / 33% headline is computed.
- **Outside the star.** `event_risk` is the Python model's output, joined 1:1 to `DimEvent`, so
  re-running the model only needs a refresh. `Metric` is a field parameter that drives the
  Overview page's metric switcher, and `_Measures` holds the 14 DAX measures. Neither needs a
  relationship.

### Cleaning (Power Query)

| Problem in the raw files | Fix |
|---|---|
| `event_id` restarts at 0 every year, so "event 5" exists ten times | `EventKey` = year + id, e.g. `2021-Texas-1` |
| Repeated DOE reports list the same county outage two or three times under one event | Remove duplicates on EventKey + FIPS + start time: **4,736 rows removed, 109,874 left** |
| One physical outage can belong to two overlapping events | `OutageID` = FIPS + start time, so measures count each outage once |
| About 60 messy event labels (`"- Severe Weather"`, `Transmision`) | Trim, fix typos, then group into **7 cause categories** (first match wins) |
| Monthly totals include month-0 yearly rows and start before the event data | Drop month 0 and cut to Nov 2014 onward so both facts cover the same months |

### The key measure

Overlapping events share county outages. A plain `SUM` over `EventOutages` counts a shared outage once
per event and gives **5.23 billion** customer-hours, about double the real total. The fix sums
distinct outages:

```dax
Customer-Hours Lost =
SUMX(
    SUMMARIZE(EventOutages, EventOutages[OutageID], EventOutages[CustomerHours]),
    EventOutages[CustomerHours]
)   -- 2.55 billion
```

The same double count shows up again later as the analyst agent's
[Texas trap](#analyst-agent-srcagentpy).

## The Power BI report

Three pages, each answering one question. Open `powerbi/grid-reliability.pbix` in Power BI Desktop;
the theme is `powerbi/simply-modern-grid.json`.

**1. Overview: how bad are outages, and where and when do they hit?**
A metric switcher (all outages, customer-weighted hours, longest single outage) drives the monthly
line. The spikes are named storms: Irma (Sep 2017), Isaias and Laura (Aug 2020), the Texas freeze
(Feb 2021), Ida (Aug 2021) and Ian (Sep 2022). Florida, Texas and California carry the most impact.

![Overview page](powerbi/screenshots/Overview.png)

**2. Causes: what causes the worst outages?**
Weather and natural disasters account for 2.45 of the 2.55 billion customer-hours (96%). All ten of the
largest events are the February 2021 Texas freeze, Hurricane Irma (2017) or Hurricane Ida (2021).

![Causes page](powerbi/screenshots/Causes.png)

**3. ML: what makes an event high-impact?**
Power BI's Key Influencers visual (describes all events at once) next to the Python model (tested on
years it never saw). Both point to weather, and to Gulf, Great Lakes and Carolina states.

![ML page](powerbi/screenshots/ML.png)

## The model

`src/outage_model.py`: logistic regression on what is known **when an event is reported**: weather or
not, month, and state. Anything measured afterwards (duration, customers out) would leak the answer.
It trains on 2014–2021 and is scored once on a held-out 2022–2023 test set.

| | Recall (high-impact) | Precision |
|---|---|---|
| Baseline (always predict "not high-impact") | 0.00 | – |
| Cross-validated, 2014–2021 | 0.78 | 0.51 |
| **Held-out test, 2022–2023** | **0.73** | **0.42** |

- **Recall over accuracy.** Missing a big event costs a utility more than a false alarm. The baseline
  is 72% accurate and catches nothing.
- **Time split, not random.** A random split mixes years and looks better than reality.
- **Drift.** High-impact events were 28% of events in 2014–2021 but 15% in 2022–2023, so a cut-off
  tuned on the old years flags too much in the new ones. That's why test precision drops.
- **Drivers:** Louisiana, Michigan, Texas and North Carolina, and weather-caused events.

## The agents

### Analyst agent (`src/agent.py`)

Ask a question in plain English. Claude writes pandas code, the program runs it and sends back the
output, and that repeats until it can answer. One tool (`run_python`), one loop, about 75 lines.
It also says when the data can't answer: asked what the outages cost in dollars, it declines.

```bash
uv run python src/agent.py "Which state lost the most customer-hours?"
```

**The Texas trap.** The first version added up customer-hours across events and answered Texas,
with about 2.2 billion: nearly the whole country's real total. Texas filed 39 overlapping events in
February 2021 alone, and they share the same county outages, so adding across events counted them
again and again. The code ran perfectly; the answer was wrong.
**The fix** was one rule in the system prompt: totals and rankings come from `outages.csv`. Now it
answers Florida (372 million), just ahead of Texas (364 million).

**Guardrails**
- You approve every script before it runs
- Each script runs as a separate process in `work/` with a 30-second timeout
- Output sent back to the model is capped at 4,000 characters
- A step limit (8) stops runaway loops
- Every step, tool call and token count is logged to `agent_log.jsonl`

**Evaluation.** `src/evals.py` re-runs the agent on 5 questions with known answers: **5/5 pass.**
Removing the `outages.csv` rule makes the Texas question fail, so the suite doubles as a regression
test for the prompt. `tests/test_tools.py` tests the tool itself: normal output, errors returned
instead of crashing, and the timeout.

**Cost.** About 2.5 API calls and $0.014 per question on Claude Sonnet 5.5, measured over 23
questions: about $7,000 a year at 500,000 questions.

### Model coach (`src/coach.py`)

An agent that tunes the outage model like a junior analyst running experiments (the
*evaluator-optimizer* pattern). It proposes a setting, code scores it with 5-fold cross-validation on
2014–2021, and it tries again: at most 8 tries, one change at a time, with a reason before each.
**The model proposes; the metric decides.**

**The 2022–2023 test years stay locked.** The agent never sees them. `coach.py final` scores the
chosen setting on them once, run by hand.

Goal: the highest recall with precision at 0.50 or above.

| # | Model | Features | Threshold | Recall | Precision | Precision ≥ 0.5? |
|---|---|---|---|---|---|---|
| 1 | logreg (baseline) | Weather, Month, State | 0.50 | 0.782 | 0.509 | ✅ |
| 2 | random forest | Weather, Month, State | 0.50 | 0.821 | 0.483 | ❌ |
| 3 | random forest | Weather, Month, State | 0.60 | 0.695 | 0.533 | ✅ |
| 4 | random forest | Weather, Month, State | 0.55 | 0.770 | 0.506 | ✅ |
| 5 | logreg | Weather, Month, State | 0.48 | 0.800 | 0.505 | ✅ |
| 6 | logreg | Weather, Month, State | 0.46 | 0.810 | 0.495 | ❌ |
| 7 | **logreg** | **Weather, Month, State** | **0.47** | **0.804** | **0.503** | ✅ **picked** |

All settings use balanced class weights. The coach picked 0.47 in 3 of 4 runs (0.48 in the other).
The random forest did worse than logistic regression at matched precision, so the simpler model
stays.

## Limitations

- **Coverage grew after 2014**, so early years undercount outages.
- **Only outages affecting 200 or more customers** are recorded in EAGLE-I.
- **Events are matched to outages with a 24-hour window**, which is a judgement call; the dataset also
  ships an 8-hour version.
- **The monthly all-outage files** (`*_group.csv`) behind the 6% / 33% headline are used by the
  Power BI report only and aren't committed here.
- **Drift:** high-impact events became rarer in 2022–2023, so precision on new data is lower than in
  cross-validation.

## Run it

```bash
uv sync
uv run python src/make_events.py      # build work/events.csv and work/outages.csv (2,683 events, 79,711 outages)
uv run python src/outage_model.py     # the model and outputs/event_risk.csv
uv run python src/agent.py "Which state lost the most customer-hours?"
uv run python src/coach.py            # the coach (final: run once, by hand)
uv run python src/evals.py            # score the analyst agent on known answers
uv run python -m pytest               # tool tests (the timeout test takes about 30 s)
```

The agents need `ANTHROPIC_API_KEY` in a `.env` file in the repo root. Every script finds its files
from the repo root, so they run from any folder.

```
data/       raw event-linked outage CSVs, one per year (2014–2023)
src/        make_events, outage_model, agent, coach, evals
tests/      tool tests
outputs/    model risk scores read by the report
powerbi/    report (.pbix), theme, screenshots
work/       generated tables, charts and scratch scripts (not committed)
```

## Data

[Event-correlated Outage Dataset in America](https://data.openei.org/submissions/6458)
(Pacific Northwest National Laboratory, on OpenEI, CC BY 4.0): EAGLE-I county outage records
(Oak Ridge National Laboratory / U.S. DOE) correlated with DOE-417 electric disturbance reports.
