# Databricks notebook source
import pandas as pd

from src.monitoring.drift_report import compute_drift_report

GOLD_TABLE = "bank_mlops.gold.features"
MONITORED_NUMERIC_COLUMNS = [
    "age", "balance", "campaign", "previous", "days_since_contact",
    "contact_recency_score", "campaign_intensity", "seasonal_conversion_prior",
]
MAJOR_SHIFT_ALERT_THRESHOLD = 0.25

gold_pdf = spark.table(GOLD_TABLE).toPandas()


reference_df = gold_pdf[gold_pdf["split"] == "train"]
current_df = gold_pdf[gold_pdf["split"] == "val"]

drift_report = compute_drift_report(reference_df, current_df, MONITORED_NUMERIC_COLUMNS)
display(drift_report)

major_shifts = drift_report[drift_report["psi"] >= MAJOR_SHIFT_ALERT_THRESHOLD]

if len(major_shifts) > 0:
    print(f"ALERT: {len(major_shifts)} column(s) show major drift:")
    print(major_shifts.to_string(index=False))
    # In production: send a Databricks SQL alert / webhook here, and/or
    # trigger the training_pipeline job's next run early rather than
    # waiting for its normal nightly schedule.
    dbutils.jobs.taskValues.set(key="drift_detected", value=True)
else:
    print("No major drift detected - all monitored columns stable.")
    dbutils.jobs.taskValues.set(key="drift_detected", value=False)