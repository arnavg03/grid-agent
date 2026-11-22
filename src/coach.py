"""Model coach: an agent that tunes the outage model. Code does the scoring; 2022-23 stays locked away."""
import json
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import agent                                   # same loop, new tool and instructions

events = pd.read_csv(agent.ROOT / "work" / "events.csv")
train = events[events["Year"] <= 2021]         # the only years the agent ever scores on
FEATURES = ["Weather", "Month", "State"]

def build(model, features, balanced):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), features)])
    weight = "balanced" if balanced else None
    if model == "logreg":
        est = LogisticRegression(max_iter=1000, class_weight=weight)
    else:
        est = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight=weight, random_state=0)
    return Pipeline([("pre", pre), ("model", est)])

def train_and_score(model, features, balanced, threshold):
    X, y = train[features], train["HighImpact"]
    proba = cross_val_predict(build(model, features, balanced), X, y, cv=5, method="predict_proba")[:, 1]
    pred = proba >= threshold
    return {"recall": round(recall_score(y, pred), 3),
            "precision": round(precision_score(y, pred, zero_division=0), 3),
            "share_flagged": round(float(pred.mean()), 3)}

# the coach's instructions.
#   - what it tunes (a classifier flagging events with 500,000+ customer-hours)
#   - what train_and_score returns (5-fold cross-validated recall/precision on 2014-2021)
#   - the baseline: logreg, all three features, balanced, threshold 0.5
#   - the goal: highest recall while precision stays at 0.50 or above
#   - the rules: at most 8 tries, one change at a time, say why before each,
#   - finish with a table of every try and a recommended setting

SYSTEM = """You tune a classifier that flags high impact events (events with >500000 customer-hours). Call train_and_score to try a setting; it returns 5 fold cross validated recall/precision on 2014-2021 events.
The baseline is logistic regression, all three features, balanced with threshold 0.5. The goal is to maximize recall while precision stays at 0.5 or above. You can try at most 8 settings, changing one at a time and
giving the rationale behind each try. Finish with a table of every try and your recommended setting."""

TOOLS = [{
    "name": "train_and_score",
    "description": "Trains and cross-validates one model setting on 2014-2021 events. "
                   "Returns recall, precision and the share of events flagged.",
    "input_schema": {"type": "object", "properties": {
        "model": {"type": "string", "enum": ["logreg", "random_forest"]},
        "features": {"type": "array", "items": {"type": "string", "enum": FEATURES}},
        "balanced": {"type": "boolean", "description": "Weight the rare high-impact class up"},
        "threshold": {"type": "number", "description": "Probability cut-off, 0.05 to 0.95"}},
        "required": ["model", "features", "balanced", "threshold"]},
}]

def run_tool(name, args):
    return json.dumps(train_and_score(args["model"], args["features"], args["balanced"], args["threshold"]))

if __name__ == "__main__":
    if sys.argv[1:] == ["final"]:              
        best = dict(model="logreg", features=FEATURES, balanced=True, threshold=0.47)  
        test = events[events["Year"] >= 2022]
        pipe = build(best["model"], best["features"], best["balanced"])
        pipe.fit(train[best["features"]], train["HighImpact"])
        pred = pipe.predict_proba(test[best["features"]])[:, 1] >= best["threshold"]
        print("2022-23 recall", round(recall_score(test["HighImpact"], pred), 3),
              "precision", round(precision_score(test["HighImpact"], pred), 3))
    else:
        print(agent.ask("Find the best setting.", system=SYSTEM, tools=TOOLS, run_tool=run_tool, max_steps=12))