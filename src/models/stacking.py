import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.evaluate import compute_classification_metrics
from src.models.preprocessing import all_model_features, build_preprocessor


def build_stacking_ensemble(scale_pos_weight: float) -> StackingClassifier:
    """
    Build a stacking ensemble model with Random Forest, XGBoost, and ANN classifiers.
    """
    
    # Define the base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', max_depth=12,n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=300, random_state=42, scale_pos_weight=scale_pos_weight, max_depth=6, learning_rate=0.05, eval_metric='logloss', n_jobs=-1)),
        ('ann', MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, random_state=42, max_iter=300))
    ]   
    
    return StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=1000),
        stacking_method='predict_proba',
        cv=5,
        n_jobs=-1
    )
    
    
def train_and_evaluate_stacking(train_df: pd.DataFrame, test_df: pd.DataFrame, freq_encoded_train: pd.DataFrame, freq_encoded_test: pd.DataFrame) -> dict:
    """
    Train and evaluate the stacking ensemble model.
    """
    
    # Build the preprocessor
    n_pos = (train_df['y'] == 1).sum()
    n_neg = (train_df['y'] == 0).sum()
    scale_pos_weight = n_neg / n_pos
    
    
    X_train, y_train = freq_encoded_train[all_model_features], freq_encoded_train['y']
    X_test, y_test = freq_encoded_test[all_model_features], freq_encoded_test['y']
    
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor(scale_numeric = False)),
        ('stacking', build_stacking_ensemble(scale_pos_weight))
    ])
    pipeline.fit(X_train, y_train)
    
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    return {"model_name": "stacked_ensemble", "pipeline": pipeline, **metrics}