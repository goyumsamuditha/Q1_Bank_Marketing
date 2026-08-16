import numpy as np
import pandas as pd

def _bucket_edges(reference_values: pd.Series, n_buckets: int = 10) -> np.ndarray:
    """
    Split the reference (training) distribution into equal-frequency
    buckets.
    """
    
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = reference_values.quantile(quantiles).values
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.unique(edges)

def population_stability_index(reference_values: pd.Series, current_values: pd.Series,
                                n_buckets: int = 10) -> float:
    """Compute PSI for one numeric column between a reference (training)
    distribution and a current (live) distribution.
    """
    edges = _bucket_edges(reference_values, n_buckets)

    ref_counts, _ = np.histogram(reference_values, bins=edges)
    cur_counts, _ = np.histogram(current_values, bins=edges)

    # convert to proportions, with a tiny floor so we never divide by/take log of zero
    ref_props = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_props = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)

    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return float(psi)

def compute_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                          numeric_columns: list[str]) -> pd.DataFrame:
    """
    Run PSI across every monitored numeric column and flag which ones
    have shifted meaningfully 
    """
    rows = []
    for col in numeric_columns:
        psi = population_stability_index(reference_df[col], current_df[col])
        if psi < 0.1:
            status = "stable"
        elif psi < 0.25:
            status = "moderate_shift"
        else:
            status = "major_shift"
        rows.append({"column": col, "psi": psi, "status": status})

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)