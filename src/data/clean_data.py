import pandas as pd


lekage_columns = ["duration"]

pdays_never_contacted_sentinel = -1

def drop_lekage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features that would not be available at prediction time.
    """
    df = df.copy()
    return df.drop(columns=[col for col in lekage_columns if col in df.columns])

def engineer_pdays_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the pdays=-1 sentinel into two honest, model-friendly columns.
    """
    df = df.copy()
    never_contacted = df["pdays"] == pdays_never_contacted_sentinel
    
    df["was_previously_contacted"] = (~never_contacted).astype(int)
    df["days_since_contact"] = df["pdays"].where(~never_contacted, 0)
    
    return df.drop(columns=["pdays"])

def winsorize_columns(df: pd.DataFrame, columnn: str, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
    """
    Clip extreme values to a percentile range instead of deleting rows
    """
    df = df.copy()
    lower_bound = df[columnn].quantile(lower_quantile)
    upper_bound = df[columnn].quantile(upper_quantile)
    df[columnn] = df[columnn].clip(lower=lower_bound, upper=upper_bound)
    return df

def cap_outliers(df: pd.DataFrame, column: str, upper_quantile: float = 0.95) -> pd.DataFrame:
    """
    Cap outliers in a column based on the specified upper limit quantile.
    """
    df = df.copy()
    upper_bound = df[column].quantile(upper_quantile)
    df[column] = df[column].clip(upper=upper_bound)
    return df

def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.
    """
    df = df.copy()
    return df.drop_duplicates()

def report_unknown_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report unknown categories in categorical columns.
    """
    categorical_columns = df.select_dtypes(include= 'object').columns
    counts = {col: int((df[col] == 'unknown').sum()) for col in categorical_columns
              if 'unknown' in df[col].unique()}
    return pd.DataFrame.from_dict(counts, orient='index', columns=['unknown_count'])

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every cleaning step in the correct order.
    """
    df = drop_lekage_columns(df)
    df = engineer_pdays_sentinel(df)
    df = winsorize_columns(df, "balance", lower_quantile=0.01, upper_quantile=0.99)
    df = cap_outliers(df, "campaign", upper_quantile=0.95)
    df = remove_duplicate_rows(df)
    return df


if __name__ == "__main__":
    from src.data.load_data import load_raw_data, encode_target

    raw = encode_target(load_raw_data("data/raw/bank-full.csv"))
    print("Unknown-value counts before cleaning:")
    print(report_unknown_categories(raw))

    cleaned = clean_pipeline(raw)
    print(f"\nRows before cleaning: {len(raw):,}")
    print(f"Rows after cleaning:  {len(cleaned):,}")
    print(f"Columns after cleaning: {list(cleaned.columns)}")    