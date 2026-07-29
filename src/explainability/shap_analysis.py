import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

def compute_shap_values(fitted_tree_model, X_sample: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values for a tree-based model
    """
    explainer = shap.TreeExplainer(fitted_tree_model)
    return explainer(X_sample)


def plot_shap_summary(shap_values: shap.Explanation, output_path: str) -> None:
    """Global feature-importance chart: which features move the
    prediction the most, on average, across all clients in the sample.
    """
    shap.summary_plot(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
def plot_shap_force_for_instance(shap_values: shap.Explanation, instance_index: int,
                                  output_path: str) -> None:
    """Local explanation for ONE client: which features pushed their
    individual prediction up or down.
    """
    shap.plots.force(shap_values[instance_index], matplotlib=True, show=False)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_shap_interactions(fitted_tree_model, X_sample: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP interaction values.
    """
    explainer = shap.TreeExplainer(fitted_tree_model)
    return explainer.shap_interaction_values(X_sample)    

def summarise_top_interaction(interaction_values: np.ndarray, feature_names: list[str],
                               feature_a: str, feature_b: str) -> float:
    """
    Return the average interaction strength between two named features,
    so you can state a plain-language finding like
    """
    idx_a = feature_names.index(feature_a)
    idx_b = feature_names.index(feature_b)
    return float(np.abs(interaction_values[:, idx_a, idx_b]).mean())    