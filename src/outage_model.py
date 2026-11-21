"""Predict whether a major outage event becomes high-impact (>= 500,000 customer-hours).
Writes outputs/event_risk.csv (EventKey, Risk). Run from the repo root: uv run python src/outage_model.py
"""
import glob
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report

files = glob.glob(str(ROOT / "data" / "eaglei_outages_with_events_*_24_hours_lag.csv"))
df_list = []
for file in files:
    year_df = pd.read_csv(file)
    year_df["Year"] = int(Path(file).name.split("_")[4])  
    df_list.append(year_df)
df = pd.concat(df_list)

# unique event key (event_id restarts every year), drop repeated rows, impact
df["EventKey"] = df["Year"].astype(str) + "-" + df["event_id"]
df = df.drop_duplicates(subset=["EventKey", "fips", "start_time"])
df["CustomerHours"] = df["mean_customers"] * df["duration"]

# one row per event
events = df.groupby("EventKey").agg({
    "Year": "first",
    "state": "first",
    "Event Type": "first",
    "Datetime Event Began": "first",
    "CustomerHours": "sum"
}).reset_index()

# features (known when the event is reported) + target
def is_weather(event_type):
    event_type = event_type.lower()
    if "weather" in event_type or "natural disaster" in event_type or "storm" in event_type:
        return "Yes"
    return "No"

events["Weather"] = events["Event Type"].apply(is_weather)
events["Month"] = pd.to_datetime(events["Datetime Event Began"]).dt.month
events["HighImpact"] = events["CustomerHours"] >= 500000

# split by time - trained on 2014-2021, tested on 2022-2023
target_column = "HighImpact"
categorical_features = ["Weather", "Month", "state"]

train_df = events[events["Year"] <= 2021]
test_df = events[events["Year"] >= 2022]

X_train = train_df[categorical_features]
y_train = train_df[target_column]
X_test = test_df[categorical_features]
y_test = test_df[target_column]

# preprocessing, baseline vs LR
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

dummy_cv_results = cross_validate(DummyClassifier(), X_train, y_train, cv=5, scoring=["accuracy", "recall"])

lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
lr_cv_results = cross_validate(lr_pipeline, X_train, y_train, cv=5, scoring=["accuracy", "recall"])

cv_eval = {
    "Dummy accuracy": dummy_cv_results["test_accuracy"].mean(),
    "Dummy recall": dummy_cv_results["test_recall"].mean(),
    "LogReg accuracy": lr_cv_results["test_accuracy"].mean(),
    "LogReg recall": lr_cv_results["test_recall"].mean()
}
print(cv_eval)

# testing
lr_pipeline.fit(X_train, y_train)
print(classification_report(y_test, lr_pipeline.predict(X_test)))

# what drives it?
coefficients = pd.DataFrame({
    "feature": lr_pipeline.named_steps["preprocessor"].get_feature_names_out(),
    "coefficient": lr_pipeline.named_steps["model"].coef_[0]
})
print(coefficients.sort_values("coefficient", ascending=False).head(5))

events["Risk"] = lr_pipeline.predict_proba(events[categorical_features])[:, 1]
events[["EventKey", "Risk"]].to_csv(ROOT / "outputs" / "event_risk.csv", index=False)
