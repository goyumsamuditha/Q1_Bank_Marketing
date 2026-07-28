import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.models.evaluate import compute_classification_metrics
from src.models.preprocessing import (all_model_features, high_cardinality_cols, low_cardinality_cols, numeric_cols, apply_frequency_encoders, build_preprocessor, fit_frequency_encoders)


random_state = 42

def get_base_model(scale_pos_weight: float) -> dict[str,tuple[object,bool]]:
    """ 
    model training function that returns a dictionary of base models and whether they require scaling of numeric features.
    """
    return {
        "Logistic Regression": (LogisticRegression(penalty='l2', class_weight="balanced", random_state=random_state, max_iter=1000), True),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=15, weights='distance'), True),
        "Random Forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced", max_depth=12, random_state=random_state, n_jobs=-1), False),
        "XGBoost": (XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, random_state=random_state, n_jobs=-1, eval_metric='logloss'), False),
        "Neural Network": (MLPClassifier(hidden_layer_sizes=(64,32), alpha=1e-3, early_stopping=True, random_state=random_state, max_iter=300), True)
    }

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to the training data to handle class imbalance.
    """
    categorical_idx = [X_train.columns.get_loc(C) for C in high_cardinality_cols + low_cardinality_cols]
    smote = SMOTENC(categorical_features=categorical_idx, random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def train_and_evaluate_model(model_name: str, estimator, needs_scaling: bool, train_df: pd.DataFrame, test_df: pd.DataFrame, use_smote: bool = False) -> dict:
    """
    Train and evaluate a model, returning evaluation metrics.
    """
    # Fit frequency encoders on training data
    freq_encoders = fit_frequency_encoders(train_df, high_cardinality_cols)
    
    # Apply frequency encoders to both training and test data
    train_df_encoded = apply_frequency_encoders(train_df, freq_encoders)
    test_df_encoded = apply_frequency_encoders(test_df, freq_encoders)

    # Separate features and target
    X_train = train_df_encoded[all_model_features]
    y_train = train_df_encoded['y']
    X_test = test_df_encoded[all_model_features]
    y_test = test_df_encoded['y']

    # Apply SMOTE if specified
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    # Build preprocessing pipeline
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor(scale_numeric=needs_scaling)),
        ('classifier', estimator)
    ])
    with mlflow.start_run(run_name=model_name):
        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5)
        
        # Compute evaluation metrics
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        mlflow.log_params({"model_name": model_name, "use_smote": use_smote, "needs_scaling": needs_scaling})
        mlflow.log_metrics(metrics)
        
        mlflow.sklearn.log_model(pipeline, model_name, artifact_path="models", skops_trusted_types= ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier",
                                                                                                     "sklearn.neural_network._stochastic_optimizers.AdamOptimizer", ])
        
    return {"model_name": model_name, "metrics": metrics, "pipeline": pipeline}

def run_all_models(train_df: pd.DataFrame, test_df: pd.DataFrame, use_smote: bool = False) -> pd.DataFrame:
    """
    Run all base models and return a DataFrame of their evaluation metrics.
    """
    n_pos = (train_df['y'] == 1).sum()
    n_neg = (train_df['y'] == 0).sum()
    scale_pos_weight = n_neg / n_pos
    
    result = []
    for model_name, (estimator, needs_scaling) in get_base_model(scale_pos_weight).items():
        print(f"Training and evaluating {model_name}...")
        model_result = train_and_evaluate_model(model_name, estimator, needs_scaling, train_df, test_df, use_smote)
        result.append({k: v for k, v in model_result.items() if k != "pipeline"})  # Exclude pipeline from the summary
        print(f'[done] {model_name} : ROC AUC = {model_result["roc_auc"]:.4f}, F1 Score = {model_result["f1_score"]:.4f} PR-AUC = {model_result["pr_auc"]:.4f}')
    
    return pd.DataFrame(result)

def save_serving_artifact(best_result : dict, train_df : pd.DataFrame, test_df : pd.DataFrame, artifact_path : str = "models") -> None:
    """
    Save the best model pipeline and frequency encoders for serving.
    """
    import pickle
    from pathlib import Path

    import joblib

    from src.models.evaluate import find_optimal_threshold
    from src.features.build_features import fit_seasonal_conversion_prior
    # Save the best model pipeline and frequency encoders for serving
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save the best model pipeline
    joblib.dump(best_result["pipeline"], f"{output_dir}/final_pipeline.joblib")
    
    
    # Save the decision threshold for serving
    y_test = test_df["y"]
    y_proba = best_result["pipeline"].predict_proba(test_df[ALL_MODEL_FEATURES])[:, 1] \
        if all(c in test_df.columns for c in ALL_MODEL_FEATURES) else None
    threshold_info = find_optimal_threshold(y_test, y_proba) if y_proba is not None else {"threshold": 0.5}
    with open(f"{output_dir}/decision_threshold.txt", "w") as f:
        f.write(str(threshold_info["threshold"]))
        
    # Save the seasonal conversion prior lookup table
    month_lookup = fit_seasonal_conversion_prior(train_df)
    month_lookup.to_csv(f"{output_dir}/seasonal_conversion_prior.csv")
    
    

    # Save the frequency encoders for serving
    freq_encoders = fit_frequency_encoders(train_df, HIGH_CARDINALITY_CATEGORICAL)
    with open(f"{output_dir}/frequency_encoders.pkl", "wb") as f:
        pickle.dump(freq_encoders, f)

    print(f"Saved serving artefacts to {output_dir}/ "
          f"(pipeline, threshold={threshold_info['threshold']:.3f}, lookups)")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from src.data.clean_data import clean_pipeline
    from src.data.load_data import encode_target, load_raw_data
    from src.features.build_features import build_features_pipeline

    raw = encode_target(load_raw_data("data/raw/bank-full.csv"))
    cleaned = clean_pipeline(raw)

    train_df, temp_df = train_test_split(cleaned, test_size=0.30, stratify=cleaned["y"], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["y"], random_state=RANDOM_STATE)

    train_feat, others = build_features_pipeline(train_df, {"val": val_df, "test": test_df})

    mlflow.set_experiment("bank-term-deposit")
    comparison = run_all_models(train_feat, others["test"])
    print("\n=== Model comparison ===")
    print(comparison.to_string(index=False))
    print("\nRe-run the winning model once more to save it for serving - "
          "see README.md 'Reproducing the full pipeline' for the one-command version.")