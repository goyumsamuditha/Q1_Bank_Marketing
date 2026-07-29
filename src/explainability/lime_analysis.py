import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


def build_lime_explainer(X_train: pd.DataFrame, categorical_feature_names: list[str],
                          class_names: list[str] = ("no", "yes")) -> LimeTabularExplainer:
    """
    Build the LIME explainer once, then reuse it for every instance
    you want to explain
    """
    categorical_indices = [X_train.columns.get_loc(c) for c in categorical_feature_names
                            if c in X_train.columns]
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        categorical_features=categorical_indices,
        class_names=list(class_names),
        mode="classification",
    )
    
def explain_borderline_instances(explainer: LimeTabularExplainer, model_predict_proba,
                                  X_test: pd.DataFrame, y_proba: np.ndarray,
                                  threshold: float, band: float = 0.05,
                                  n_examples: int = 5) -> list[dict]:
    """Find clients whose predicted probability sits close to the
    decision threshold
    """
    is_borderline = np.abs(y_proba - threshold) <= band
    borderline_indices = np.where(is_borderline)[0][:n_examples]

    explanations = []
    for idx in borderline_indices:
        exp = explainer.explain_instance(
            X_test.iloc[idx].values, model_predict_proba, num_features=8
        )
        explanations.append({
            "row_index": int(idx),
            "predicted_probability": float(y_proba[idx]),
            "top_features": exp.as_list(),
        })
    return explanations   
