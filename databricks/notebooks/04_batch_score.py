# Databricks notebook source
import mlflow
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
GOLD_TABLE = "bank_mlops.gold.features"
SCORED_TABLE = "bank_mlops.gold.scored_leads"
CATALOG_MODEL_NAME = "bank_mlops.models.term_deposit_model"
DECISION_THRESHOLD = 0.32 

champion_model_uri = f"models:/{CATALOG_MODEL_NAME}@champion"
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=champion_model_uri, result_type="double")

features_df = spark.table(GOLD_TABLE)

feature_columns = [c for c in features_df.columns if c not in ("split", "client_index", "y")]

scored_df = features_df.withColumn(
    "subscription_probability", predict_udf(*[F.col(c) for c in feature_columns])
).withColumn(
    "predicted_label", F.when(F.col("subscription_probability") >= DECISION_THRESHOLD, 1).otherwise(0)
).withColumn(
    "scored_at", F.current_timestamp()
)

(
    scored_df.select("client_index", "subscription_probability", "predicted_label", "scored_at")
    .write.format("delta")
    .mode("overwrite")
    .saveAsTable(SCORED_TABLE)
)

print(f"Wrote {scored_df.count():,} scored leads to {SCORED_TABLE}")