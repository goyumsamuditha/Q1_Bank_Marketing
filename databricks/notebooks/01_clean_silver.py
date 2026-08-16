# Databricks notebook source
# MAGIC %md
# MAGIC ## 01 - Clean Silver
# MAGIC Reads the Bronze table, applies the SAME cleaning rules as
# MAGIC `src/data/clean_data.py` (converted to Spark), and writes a Silver
# MAGIC Delta table. Silver = "cleaned, but not yet feature-engineered".
# MAGIC
# MAGIC This notebook re-implements the pandas logic in `src/data/clean_data.py`
# MAGIC using PySpark, because Bronze/Silver/Gold tables are Spark DataFrames,
# MAGIC not pandas DataFrames, at Databricks scale. The RULES are identical -
# MAGIC only the DataFrame API differs. If you change a cleaning rule, change
# MAGIC it in BOTH `src/data/clean_data.py` (used by the local/pandas path)
# MAGIC and here (used by the Databricks/Spark path).

# COMMAND ----------

from pyspark.sql import functions as F

BRONZE_TABLE = "bank_mlops.bronze.clients_raw"
SILVER_TABLE = "bank_mlops.silver.clients_cleaned"

PDAYS_NEVER_CONTACTED_SENTINEL = -1

# COMMAND ----------

df = spark.table(BRONZE_TABLE)

# --- 1. Drop leakage column ---
# `duration` is only known after a call ends - never available at
# prediction time, so it must never reach the model.
df = df.drop("duration")

# --- 2. Turn the pdays=-1 sentinel into two honest columns ---
df = (
    df.withColumn(
        "was_previously_contacted",
        F.when(F.col("pdays") == PDAYS_NEVER_CONTACTED_SENTINEL, 0).otherwise(1),
    )
    .withColumn(
        "days_since_contact",
        F.when(F.col("pdays") == PDAYS_NEVER_CONTACTED_SENTINEL, 0).otherwise(F.col("pdays")),
    )
    .drop("pdays")
)

# --- 3. Winsorise balance at 1st/99th percentile ---
lower_balance, upper_balance = df.approxQuantile("balance", [0.01, 0.99], 0.001)
df = df.withColumn(
    "balance",
    F.when(F.col("balance") < lower_balance, lower_balance)
    .when(F.col("balance") > upper_balance, upper_balance)
    .otherwise(F.col("balance")),
)

# --- 4. Cap campaign at the 95th percentile ---
upper_campaign = df.approxQuantile("campaign", [0.95], 0.001)[0]
df = df.withColumn(
    "campaign", F.when(F.col("campaign") > upper_campaign, upper_campaign).otherwise(F.col("campaign"))
)

# --- 5. Encode target as numeric ---
df = df.withColumn("y", F.when(F.col("y") == "yes", 1).otherwise(0))

# --- 6. Drop exact duplicate rows ---
before = df.count()
df = df.dropDuplicates()
after = df.count()
print(f"Removed {before - after} duplicate rows")

# COMMAND ----------

df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(SILVER_TABLE)
print(f"Wrote Silver table: {SILVER_TABLE} ({after:,} rows)")
