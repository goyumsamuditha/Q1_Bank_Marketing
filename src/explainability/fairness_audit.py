import pandas as pd

from src.models.evaluate import compute_classification_metrics

def subgroup_performance(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                          y_proba_col: str, sensitive_column: str) -> pd.DataFrame:
    """
    Compute the six classification metrics separately for every value
    of a sensitive/protected-proxy column 
    """
    rows = []
    for group_value, group_df in df.groupby(sensitive_column, observed=True):
        if group_df[y_true_col].nunique() < 2:
            # Skip groups with only one class present
            continue
         metrics = compute_classification_metrics(
            group_df[y_true_col], group_df[y_pred_col], group_df[y_proba_col]
        )
        rows.append({sensitive_column: group_value, "n_clients": len(group_df), **metrics})

    overall_metrics = compute_classification_metrics(df[y_true_col], df[y_pred_col], df[y_proba_col])
    rows.append({sensitive_column: "OVERALL", "n_clients": len(df), **overall_metrics})

    return pd.DataFrame(rows)

def flag_disparate_subgroups(subgroup_table: pd.DataFrame, metric: str = "recall",
                              tolerance: float = 0.10) -> pd.DataFrame:
    """Flag any subgroup whose chosen metric is more than `tolerance`
    below the overall value - a simple, explainable disparate-impact
    check 
    """
    
    overall_value = subgroup_table.loc[
        subgroup_table.iloc[:, 0] == "OVERALL", metric
    ].iloc[0]

    flagged = subgroup_table[(subgroup_table.iloc[:, 0] != "OVERALL") &(subgroup_table[metric] < overall_value - tolerance)].copy()
    flagged["gap_vs_overall"] = overall_value - flagged[metric]
    return flagged  