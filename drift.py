# drift.py

import pandas as pd

from ingest import load_data, merge_data

from evidently import Report
from evidently.presets import DataDriftPreset


def split_data(df, split_ratio=0.8):
    split = int(len(df) * split_ratio)
    return df.iloc[:split], df.iloc[split:]


def run_drift_report(old_data, new_data):

    old_data = old_data.select_dtypes(include=["number"])
    new_data = new_data.select_dtypes(include=["number"])

    report = Report(metrics=[DataDriftPreset()])

    my_eval = report.run(
        reference_data=old_data,
        current_data=new_data
    )

    my_eval.json()
    my_eval.dict()
    my_eval.save_html("New_file.html")

    print("📊 Drift report saved as New_file.html")


if __name__ == "__main__":
    train, test, store = load_data()
    df = merge_data(train, store)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"])

    old_data, new_data = split_data(df)

    run_drift_report(old_data, new_data)