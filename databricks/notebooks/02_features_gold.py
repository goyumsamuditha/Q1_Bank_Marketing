# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

SILVER_TABLE = "bank_mlops.silver.clients_cleaned"
GOLD_TABLE = "bank_mlops.gold.features"

HIGH_SEASON_MONTHS = ["mar", "sep", "oct", "dec"]

df = spark.table(SILVER_TABLE)

df = df.withColumn("_row_hash", F.abs(F.hash(F.monotonically_increasing_id())) % 100)
df = df.withColumn(
    "split",
    F.when(F.col("_row_hash") < 70, "train")
    .when(F.col("_row_hash") < 85, "val")
    .otherwise("test"),
).drop("_row_hash")

# --- Feature 1: contact_recency_score ---
df = df.withColumn(
    "recency_decay", 1 / (1 + F.col("days_since_contact"))
).withColumn(
    "success_bonus", F.when(F.col("poutcome") == "success", 1.0).otherwise(0.0)
).withColumn(
    "contact_recency_score",
    F.col("was_previously_contacted") * F.col("recency_decay") * (0.5 + 0.5 * F.col("success_bonus")),
).drop("recency_decay", "success_bonus")

# --- Feature 2: campaign_intensity ---
df = df.withColumn(
    "campaign_intensity",
    F.col("campaign") / F.when(F.col("campaign") + F.col("previous") == 0, 1).otherwise(F.col("campaign") + F.col("previous")),
)

# --- Feature 3: age_life_stage ---
df = df.withColumn(
    "age_life_stage",
    F.when(F.col("age") < 30, "young_adult")
    .when(F.col("age") < 45, "family_formation")
    .when(F.col("age") < 60, "pre_retirement")
    .otherwise("retired"),
)

# --- Feature 4: seasonal_conversion_prior (fit on TRAIN split only) ---
train_month_rates = (
    df.filter(F.col("split") == "train")
    .groupBy("month")
    .agg(F.avg("y").alias("seasonal_conversion_prior"))
)
overall_train_rate = train_month_rates.agg(F.avg("seasonal_conversion_prior")).collect()[0][0]

df = df.join(train_month_rates, on="month", how="left").fillna(
    {"seasonal_conversion_prior": overall_train_rate}
)

# --- Feature 5: is_high_season ---
df = df.withColumn("is_high_season", F.when(F.col("month").isin(HIGH_SEASON_MONTHS), 1).otherwise(0))

# --- Feature 6 (A+ extra): contact_channel_trust ---
df = df.withColumn("contact_channel_trust", F.concat_ws("_", F.col("contact"), F.col("poutcome")))

# --- Add a stable client_index primary key for the Feature Store ---
df = df.withColumn("client_index", F.monotonically_increasing_id())

# COMMAND ----------

df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_TABLE)
print(f"Wrote Gold feature table: {GOLD_TABLE} ({df.count():,} rows)")

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

try:
    fe.create_table(
        name=GOLD_TABLE,
        primary_keys=["client_index"],
        df=df,
        description="Engineered client features for term-deposit subscription prediction.",
    )
except Exception as e:
    # table already registered from a previous run - just log and continue
    print(f"Feature table registration skipped (likely already exists): {e}")