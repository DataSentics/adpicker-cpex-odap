# Databricks notebook source
# MAGIC %md ###Try to connect ot `feature store - latest` from different workspaces

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

dbutils.secrets.list("adls-gen2-prod")

# COMMAND ----------

APP_ENV = "prod"

# COMMAND ----------

path_to_fs = f"abfss://gold@cpexstorageblob{APP_ENV}.dfs.core.windows.net/feature_store/features/user_entity.delta"
df = spark.read.format("delta").load(path_to_fs)
display(df)
