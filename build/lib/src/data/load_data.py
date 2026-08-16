from pathlib import Path
import pandas as pd

csv_separator = ";"

raw_columns = [ "age", "job", "marital", "education", "default", "balance",
                "housing", "loan", "contact", "day", "month", "duration",
                "campaign", "pdays", "previous", "poutcome", "y"]

def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Load the raw data from a CSV file.
    """
    
    df = pd.read_csv(path, sep=csv_separator)
    
    missing_columns = set(raw_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in the CSV file: {missing_columns}")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the text target 'yes'/'no' into a numeric 0/1 column.
    """
    df = df.copy()
    df["y"] = df["y"].map({"yes": 1, "no": 0}).astype(int)
    return df

if __name__ == "__main__":
    raw = load_raw_data("data/raw/bank-full.csv")
    raw = encode_target(raw)
    print(f"Loaded {raw.shape[0]:,} rows, {raw.shape[1]} columns")
    print(f"Subscription rate: {raw['y'].mean():.2%}")   