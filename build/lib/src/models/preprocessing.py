import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

low_cardinality_cols = ["marital", "contact", "poutcome", "housing", "loan", "default","age_life_stage", "contact_channel_trust",]

high_cardinality_cols = ["job", "education", "month"]

numeric_cols = ["age", "balance", "day", "campaign", "previous","was_previously_contacted", "days_since_contact","contact_recency_score", 
                "campaign_intensity","seasonal_conversion_prior", "is_high_season",]

high_cardinality_frequency_cols = [f'{c}_freq' for c in high_cardinality_cols]

all_model_features = low_cardinality_cols + high_cardinality_frequency_cols + numeric_cols

def fit_frequency_encoders(train_df: pd.DataFrame, columns: list[str]) -> dict[str, pd.Series]:
    """
    Fit frequency encoders for high cardinality categorical features.
    """
    freq_encoders = {}
    for col in columns:
        freq_encoders[col] = train_df[col].value_counts(normalize=True)
    return freq_encoders

def apply_frequency_encoders(df: pd.DataFrame, freq_encoders: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Apply previously fit frequency encoders to any split.
    """
    df_encoded = df.copy()
    for col, freq_encoder in freq_encoders.items():
        df_encoded[f'{col}_freq'] = df_encoded[col].map(freq_encoder).fillna(0.0)
    return df_encoded

def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """
    Build a preprocessor for the model pipeline.
    """
    numeric_transformer = StandardScaler() if scale_numeric else 'passthrough'
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols + high_cardinality_frequency_cols),
            ('low_card', OneHotEncoder(handle_unknown='ignore'), low_cardinality_cols),
        ],
        remainder='drop'
    )