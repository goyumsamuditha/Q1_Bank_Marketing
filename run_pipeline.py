import mlflow
from sklearn.model_selection import train_test_split

from src.data.clean_data import clean_pipeline
from src.data.load_data import encode_target, load_raw_data
from src.features.build_features import build_features_pipeline
from src.models.evaluate import final_optimal_threshold
from src.models.preprocessing import all_model_features
from src.models.stacking import train_and_evaluate_stacking
from src.models.train import run_all_models, save_serving_artifact, fit_frequency_encoders
from src.models.preprocessing import high_cardinality_cols, apply_frequency_encoders

RANDOM_STATE = 42
RAW_DATA_PATH = "data/raw/bank-full.csv"


def main() -> None:
    print("Step 1/6: loading and cleaning data...")
    raw = encode_target(load_raw_data(RAW_DATA_PATH))
    cleaned = clean_pipeline(raw)
    print(f"  -> {len(cleaned):,} rows after cleaning")

    print("Step 2/6: splitting train/val/test (70/15/15, stratified)...")
    train_df, temp_df = train_test_split(
        cleaned, test_size=0.30, stratify=cleaned["y"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["y"], random_state=RANDOM_STATE
    )

    print("Step 3/6: engineering features (train-only fits applied to val/test)...")
    train_feat, others = build_features_pipeline(train_df, {"val": val_df, "test": test_df})

    print("Step 4/6: training all 5 required models + logging to MLflow...")
    mlflow.set_experiment("bank-term-deposit")
    comparison = run_all_models(train_feat, others["test"])
    print(comparison.to_string(index=False))

    print("Step 5/6: training the stacked ensemble (A+ extra 6th model)...")
    freq_encoders = fit_frequency_encoders(train_feat, high_cardinality_cols)
    train_enc = apply_frequency_encoders(train_feat, freq_encoders)
    test_enc = apply_frequency_encoders(others["test"], freq_encoders)
    stack_result = train_and_evaluate_stacking(train_feat, others["test"], train_enc, test_enc)
    print(f"  -> stacked_ensemble: ROC-AUC={stack_result['roc_auc']:.3f} PR-AUC={stack_result['pr_auc']:.3f}")

    print("Step 6/6: saving serving artefacts for the best model...")
    all_results = comparison.to_dict("records") + [
        {k: v for k, v in stack_result.items() if k != "pipeline"}
    ]
    best_name = max(all_results, key=lambda r: r["pr_auc"])["model_name"]
    print(f"  -> best model by PR-AUC: {best_name}")

 
    if best_name == "stacked_ensemble":
        best_pipeline_result = stack_result
    else:
        from src.models.train import get_base_model, train_and_evaluate_model
        n_pos = (train_feat["y"] == 1).sum()
        n_neg = (train_feat["y"] == 0).sum()
        estimator, needs_scaling = get_base_model(n_neg / n_pos)[best_name]
        best_pipeline_result = train_and_evaluate_model(
            best_name, estimator, needs_scaling, train_feat, others["test"]
        )

    save_serving_artifact(best_pipeline_result, train_feat, others["test"])
    print("\nDone. Run `uvicorn src.serving.api:app --reload` to serve predictions locally.")


if __name__ == "__main__":
    main()
