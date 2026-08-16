import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = 'roc_auc'  # Use AUC as the scoring metric

def tune_logistic_regression(X_train, y_train):
    """ Tune hyperparameters for Logistic Regression using GridSearchCV.
    """
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        "class_weight" : ["balanced"]
    }

    search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid, scoring=scoring, cv=cv_folds, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_

def tune_random_forest(X_train, y_train, n_iter: int = 25):
    """ Tune hyperparameters for Random Forest using RandomizedSearchCV.
    """
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        param_distributions, n_iter=n_iter, scoring=scoring, cv=cv_folds, n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_

def tune_xgboost(X_train, y_train, scale_pos_weight: float, n_trials: int = 30):
    """ Tune hyperparameters for XGBoost using Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            **params
        )
        scores = []
        for train_idx, val_idx in cv_folds.split(X_train, y_train):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            y_pred_proba = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            from sklearn.metrics import roc_auc_score
            scores.append(roc_auc_score(y_train.iloc[val_idx], y_pred_proba))
        return sum(scores) / len(scores)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **study.best_params
    )
    best_model.fit(X_train, y_train)
    return best_model, study.best_params, study.best_value

def tune_knn(X_train, y_train):
    """ Tune hyperparameters for K-Nearest Neighbors using GridSearchCV.
    """
    param_grid = {
        "n_neighbors": [5, 10, 15, 25, 40],
        "metric": ["euclidean", "manhattan"],
        "weights": ["uniform", "distance"]
    }
    
    search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid, scoring=scoring, cv=cv_folds, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_

def tune_ann(X_train, y_train, n_iter: int = 15):
    """ Tune hyperparameters for Artificial Neural Network using RandomizedSearchCV.
    """
    param_distributions = {
        "hidden_layer_sizes": [(32,), (64, 32), (128, 64), (64, 32, 16)],
        "alpha": [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [0.0005, 0.001, 0.005],
        "batch_size": [64, 128, 256]
    }
    search = RandomizedSearchCV(
        MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
        param_distributions, n_iter=n_iter, scoring=scoring, cv=cv_folds, n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_