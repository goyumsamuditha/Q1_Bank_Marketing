import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)


def compute_classification_metrics(y_true, y_pred, y_proba) -> dict:
    """
    Compute classification metrics for model evaluation.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba)
    }
    
    return metrics

def get_confusion_matrix(y_true, y_pred) -> dict:
    """
    Compute confusion matrix for model evaluation.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    matrix = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }
    
    return matrix

def final_optimal_threshold(y_true, y_proba, metric: str = "f1_score", min_recall: float | None = None) -> float:
    """
    Find the optimal threshold for classification based on a specified metric.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best = {"threshold": 0.5, "f1_score": -1}
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        if min_recall is not None and recall < min_recall:
            continue
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best["f1_score"]:
            best = {"threshold": float(threshold), "f1_score": f1, "recall": recall, "precision": precision_score(y_true, y_pred, zero_division=0)}
            
    return best


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """
    Build a comparison table from a list of results dictionaries.
    """
    df = pd.DataFrame(results)
    return df.sort_values("pr_auc", ascending=False).reset_index(drop=True)