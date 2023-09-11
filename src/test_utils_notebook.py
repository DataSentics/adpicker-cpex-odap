# Databricks notebook source
import pyspark.sql.functions as F

# COMMAND ----------

meta = spark.read.format("delta").load("abfss://gold@cpexstorageblobdev.dfs.core.windows.net/feature_store/metadata/metadata.delta")
meta.filter(F.col("category") == "digital_general").display()

# COMMAND ----------

spark.read.table("odap_features_user.metadata").filter(F.col("category") == "digital_device").display()

# COMMAND ----------

df_interest = spark.read.format("delta").load("/mnt/aam-cpex-dev/silver/interests/interests_definition.delta")
df_interest.display()


# COMMAND ----------


