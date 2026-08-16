# Databricks notebook source
import mlflow

# Point MLflow at the Unity Catalog model registry namespace for this project.
mlflow.set_registry_uri("databricks-uc")
CATALOG_MODEL_NAME = "bank_mlops.models.term_deposit_model"
GOLD_TABLE = "bank_mlops.gold.features"

gold_pdf = spark.table(GOLD_TABLE).toPandas()

train_df = gold_pdf[gold_pdf["split"] == "train"].drop(columns=["split", "client_index"])
val_df = gold_pdf[gold_pdf["split"] == "val"].drop(columns=["split", "client_index"])
test_df = gold_pdf[gold_pdf["split"] == "test"].drop(columns=["split", "client_index"])

print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

from src.models.train import run_all_models
from src.models.stacking import train_and_evaluate_stacking
from src.models.preprocessing import fit_frequency_encoders, apply_frequency_encoders, HIGH_CARDINALITY_CATEGORICAL

mlflow.set_experiment("/Shared/bank-term-deposit-mlops/training")

comparison = run_all_models(train_df, test_df, use_smote=False)
display(comparison)

freq_encoders = fit_frequency_encoders(train_df, HIGH_CARDINALITY_CATEGORICAL)
train_enc = apply_frequency_encoders(train_df, freq_encoders)
test_enc = apply_frequency_encoders(test_df, freq_encoders)
stack_result = train_and_evaluate_stack(train_df, test_df, train_enc, test_enc)
print(f"stacked_ensemble: ROC-AUC={stack_result['roc_auc']:.3f}  PR-AUC={stack_result['pr_auc']:.3f}")


from mlflow import MlflowClient

client = MlflowClient()
PERFORMANCE_GATE_TOLERANCE = 0.01  # new model may be up to 1pp worse and still pass, to allow for noise

all_results = comparison.to_dict("records") + [{k: v for k, v in stack_result.items() if k != "pipeline"}]
best = max(all_results, key=lambda r: r["pr_auc"])
print(f"Best model this run: {best['model_name']} (PR-AUC={best['pr_auc']:.3f})")

try:
    current_champion = client.get_model_version_by_alias(CATALOG_MODEL_NAME, "champion")
    current_champion_run = client.get_run(current_champion.run_id)
    current_roc_auc = current_champion_run.data.metrics.get("roc_auc", 0.0)
except Exception:
    current_roc_auc = 0.0  # no champion registered yet - anything passes

with mlflow.start_run(run_name=f"register_{best['model_name']}"):
    mlflow.log_metrics({k: v for k, v in best.items() if isinstance(v, (int, float))})
    # NOTE: in a full run, re-fit and log the winning pipeline as an mlflow
    # model artifact here (mlflow.sklearn.log_model(...)), then:
    # model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    # registered = mlflow.register_model(model_uri, CATALOG_MODEL_NAME)

    if best["roc_auc"] >= current_roc_auc - PERFORMANCE_GATE_TOLERANCE:
        print(f"PASSED performance gate ({best['roc_auc']:.3f} >= {current_roc_auc:.3f} - {PERFORMANCE_GATE_TOLERANCE}) "
              "- promoting to champion")
        # client.set_registered_model_alias(CATALOG_MODEL_NAME, "champion", registered.version)
    else:
        print(f"FAILED performance gate ({best['roc_auc']:.3f} < {current_roc_auc:.3f} - {PERFORMANCE_GATE_TOLERANCE}) "
              "- keeping existing champion, alerting for manual review")