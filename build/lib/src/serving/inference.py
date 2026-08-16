from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import (
    add_age_life_stage,
    add_campaign_intensity,
    add_contact_channel_trust,
    add_contact_recency_score,
    add_is_high_season,
)
from src.models.preprocessing import ALL_MODEL_FEATURES, apply_frequency_encoders


def load_model(model_path: str | Path):
    """Load a model from a given path."""
    return joblib.load(model_path)

def prepare_single_record(record: dict, seasonal_prior_lookup: pd.Series,
                           frequency_encoders: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Apply the same feature-engineering steps used in training to one
    incoming client record
    """
    
    df = pd.DataFrame([record])

    never_contacted = df["pdays"] == -1
    df["was_previously_contacted"] = (~never_contacted).astype(int)
    df["days_since_contact"] = df["pdays"].where(~never_contacted, 0)
    df = df.drop(columns=["pdays"])

    df = add_contact_recency_score(df)
    df = add_campaign_intensity(df)
    df = add_age_life_stage(df)
    df = add_is_high_season(df)
    df = add_contact_channel_trust(df)

    overall_mean = seasonal_prior_lookup.mean()
    df["seasonal_conversion_prior"] = df["month"].map(seasonal_prior_lookup).fillna(overall_mean)

    df = apply_frequency_encoders(df, frequency_encoders)

    return df

def predict_single(pipeline, prepared_row: pd.DataFrame, threshold: float) -> dict:
    """
    Run the fitted pipeline on one prepared row and apply the tuned
    decision threshold
    """
    proba = float(pipeline.predict_proba(prepared_row[ALL_MODEL_FEATURES])[0, 1])
    label = "yes" if proba >= threshold else "no"
    return {
        "subscription_probability": proba,
        "decision_threshold_used": threshold,
        "predicted_label": label,
    }
    

def predict_batch(pipeline, prepared_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Same as predict_single but for a whole DataFrame at once
    """
    proba = pipeline.predict_proba(prepared_df[ALL_MODEL_FEATURES])[:, 1]
    result = prepared_df.copy()
    result["subscription_probability"] = proba
    result["predicted_label"] = (proba >= threshold).astype(int)
    return result