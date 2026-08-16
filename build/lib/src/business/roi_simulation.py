import numpy as np
import pandas as pd

default_cost_per_call = 8.0
default_revenue_per_conversion = 120.0

def simulate_net_value_by_threshold(y_true: np.ndarray, y_proba: np.ndarray, cost_per_call: float = default_cost_per_call,
                                     revenue_per_conversion: float = default_revenue_per_conversion,thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """
    Simulate the net value of a marketing campaign based on predicted probabilities and a range of thresholds.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        true_positives = int(((y_pred == 1) & (y_true == 1)).sum())
        total_calls_made = int((y_pred == 1).sum())

        revenue = true_positives * revenue_per_conversion
        cost = total_calls_made * cost_per_call
        net_value = revenue - cost

        rows.append({
            "threshold": t,
            "calls_made": total_calls_made,
            "conversions": true_positives,
            "revenue": revenue,
            "cost": cost,
            "net_value": net_value,
        })

    return pd.DataFrame(rows)

def find_roi_optimal_threshold(roi_table: pd.DataFrame) -> dict:
    """Pick the threshold that maximises net_value from the table above."""
    best_row = roi_table.loc[roi_table["net_value"].idxmax()]
    return best_row.to_dict()


def compare_f1_vs_roi_thresholds(y_true: np.ndarray, y_proba: np.ndarray, f1_optimal_threshold: float,
                                  cost_per_call: float = default_cost_per_call,revenue_per_conversion: float = default_revenue_per_conversion) -> pd.DataFrame:
    """
    Compare the F1-optimal threshold with the ROI-optimal threshold.
    """
    roi_table = simulate_net_value_by_threshold(y_true, y_proba, cost_per_call, revenue_per_conversion)
    roi_optimal = find_roi_optimal_threshold(roi_table)

    f1_row = roi_table.iloc[(roi_table["threshold"] - f1_optimal_threshold).abs().argsort().iloc[0]]

    return pd.DataFrame([
        {"strategy": "F1-optimal", **f1_row.to_dict()},
        {"strategy": "ROI-optimal", **roi_optimal},
    ])