RAW_CSV_PATH = "abfss://landing@<your-storage-account>.dfs.core.windows.net/bank-full.csv"
BRONZE_TABLE = "bank_mlops.bronze.clients_raw"

# COMMAND ----------

df = (
    spark.read.option("header", True)
    .option("sep", ";")
    .option("inferSchema", True)
    .csv(RAW_CSV_PATH)
)

print(f"Ingested {df.count():,} rows, {len(df.columns)} columns")
display(df.limit(5))

# COMMAND ----------

(
    df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_TABLE)
)

print(f"Wrote Bronze table: {BRONZE_TABLE}")
