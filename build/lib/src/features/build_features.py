import pandas as pd

age_life_stage_bins = [0,30,45,60,200]
age_life_stage_labels = ["Young Adult", "family_formation", "pre_retirement", "retired"]

high_season_months = {"mar", "sep","oct", "dec"}

def add_contact_recency_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 1: how recently AND how successfully was this client last contacted?
    """
    df = df.copy()
    
    recency_decay = 1/(1 + df["days_since_contact"])
    
    success_bonus = (df["poutcome"] == "success").astype(float)
    
    base_score = 0.5 + 0.5 * success_bonus  
    
    score = df["was_previously_contacted"] * recency_decay * base_score
    df["contact_recency_score"] = score
    return df

def add_campaign_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 2: what fraction of this client's total contacts happened
    in THIS campaign, versus previous campaigns?
    """
    df = df.copy()
    total_contacts = df["campaign"] + df["previous"]
    
    df["campaign_intensity"] = df["campaign"] / total_contacts.replace(0, 1)  # Avoid division by zero
    return df

def add_age_life_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 3: bucket age into life stages.
    """
    df = df.copy()
    df["age_life_stage"] = pd.cut(df["age"], bins=age_life_stage_bins, labels=age_life_stage_labels, right=False).astype(str)   
    return df


def fit_seasonal_conversion_prior(train_df: pd.DataFrame) -> pd.Series:
    """
    Fit step for Feature 4: average historical subscription rate per month,
    computed ONLY on the training split.
    """
    return train_df.groupby("month")["y"].mean()


def apply_seasonal_conversion_prior(df: pd.DataFrame, month_lookup: pd.Series) -> pd.DataFrame:
    """Apply step for Feature 4: map each row's month to the training-set
    subscription rate for that month. Unseen months (shouldn't happen with
    12 possible months, but just in case) fall back to the overall mean.
    """
    df = df.copy()
    overall_mean = month_lookup.mean()
    df["seasonal_conversion_prior"] = df["month"].map(month_lookup).fillna(overall_mean)
    return df

def add_is_high_season(df: pd.DataFrame) -> pd.DataFrame:
    """Feature 5 (bonus): binary flag for the months your EDA found convert
    far better (March, September, October, December).
    """
    df = df.copy()
    df["is_high_season"] = df["month"].isin(high_season_months).astype(int)
    return df

def add_contact_channel_trust(df: pd.DataFrame) -> pd.DataFrame:
    """Feature 6 (A+ extra): interaction between contact channel and past
    campaign outcome.
    """
    df = df.copy()
    df["contact_channel_trust"] = df["contact"].astype(str) + "_" + df["poutcome"].astype(str)
    return df

def build_features_pipeline(train_df : pd.DataFrame, other_splits: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run every feature-engineering step on the training split, and apply
    """
    other_splits = other_splits or {}
    
    def add_leak_free_features(df: pd.DataFrame) -> pd.DataFrame:
        df = add_contact_recency_score(df)
        df = add_campaign_intensity(df)
        df = add_age_life_stage(df)
        df = add_is_high_season(df)
        df = add_contact_channel_trust(df)
        return df
    
    train_out = add_leak_free_features(train_df)
    
    month_lookup = fit_seasonal_conversion_prior(train_out)
    train_out = apply_seasonal_conversion_prior(train_out, month_lookup)
    
    other_out = {}
    for name, split_df in other_splits.items():
        split_out = add_leak_free_features(split_df)
        split_out = apply_seasonal_conversion_prior(split_out, month_lookup)
        other_out[name] = split_out
    
    return train_out, other_out

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from src.data.load_data import load_raw_data, encode_target
    from src.data.clean_data import clean_pipeline
    
    raw = encode_target(load_raw_data("data/raw/bank-full.csv"))
    cleaned = clean_pipeline(raw)
    
    train_df, temp_df = train_test_split(cleaned, test_size=0.30, stratify=cleaned["y"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["y"], random_state=42)
    
    train_features, other_features = build_features_pipeline(train_df, {"val": val_df, "test": test_df})
    print("New Columns added:", [col for col in train_features.columns if col not in train_df.columns])
    print(train_features[["contact_recency_score", "campaign_intensity", "age_life_stage",
                       "seasonal_conversion_prior", "is_high_season"]].head())