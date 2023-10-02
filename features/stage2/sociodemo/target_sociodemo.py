# Databricks notebook source
# MAGIC %md #### Imports

# COMMAND ----------

import pyspark.sql.functions as F

from pyspark.sql.dataframe import DataFrame

from src.utils.helper_functions_defined_by_user.yaml_functions import get_value_from_yaml

# COMMAND ----------

# MAGIC %md
# MAGIC #### Config

# COMMAND ----------

PUBLISHERS_LIST = ['eco']

# COMMAND ----------

dbutils.widgets.text("timestamp", "")

# COMMAND ----------

timestamp = dbutils.widgets.get("timestamp")

# COMMAND ----------

# MAGIC %md #### Load inputs

# COMMAND ----------

# MAGIC %md Load sociodemo data

# COMMAND ----------

df_sdm_sociodemo_targets = spark.read.format("delta").load(
    get_value_from_yaml("paths", "sdm_sociodemo_targets")
)

# COMMAND ----------

# MAGIC %md Load feature store

# COMMAND ----------

def read_fs(timestamp):
    df_fs = (
        spark.read.table("odap_features_user.user_stage1")
        .filter(F.col("timestamp") == timestamp)
        .select("user_id", "timestamp")
        .withColumn("timestamp", F.to_timestamp(F.col("timestamp")))
    )
    return df_fs


df_fs = read_fs(timestamp)

# COMMAND ----------

# MAGIC %md #### Processing

# COMMAND ----------

def process_publishers(df: DataFrame, publishers_list):
    if len(publishers_list) > 0:
        df = df.withColumn(
            "AGE",
            F.when(F.col("PUBLISHER").isin(publishers_list), F.col("AGE")).otherwise(
                "unknown"
            ),
        ).withColumn(
            "GENDER",
            F.when(F.col("PUBLISHER").isin(publishers_list), F.col("GENDER")).otherwise(
                "unknown"
            ),
        )

    return df


df_process_publishers = process_publishers(df_sdm_sociodemo_targets, PUBLISHERS_LIST)

# COMMAND ----------

def process_age_categories(df: DataFrame):
    return (df
            .withColumn('AGE', F.col('AGE').cast('integer'))
            .withColumn('LABEL_AGE',
                        F.when(F.col('AGE').between(0, 17), '0')
                        .when(F.col('AGE').between(18, 24), '1')
                        .when(F.col('AGE').between(25, 34), '2')
                        .when(F.col('AGE').between(35, 44), '3')
                        .when(F.col('AGE').between(45, 54), '4')
                        .when(F.col('AGE').between(55, 64), '5')
                        .when(F.col('AGE').between(65, 100), '6')
                        .otherwise('unknown')
                      )
            .withColumn('GENDER', F.when(F.col('GENDER') == "1", "0")
                                  .when(F.col('GENDER') == "2", "1")
                       )
           )
    
df_process_age_categories = process_age_categories(df_process_publishers)

# COMMAND ----------

def aggregate_targets(df: DataFrame, df_fs: DataFrame):
    df = (df
          .groupBy("USER_ID")
          .agg(
              *[F.array_distinct(F.collect_set(x)).alias(x) for x in ["LABEL_AGE", "GENDER"]],
          )
          .withColumn("LABEL_AGE", F.array_remove(F.col("LABEL_AGE"), "unknown"))
          .withColumn("GENDER", F.array_remove(F.col("GENDER"), "unknown"))
          .withColumn("AGE_SIZE", F.size("LABEL_AGE"))
          .withColumn("GENDER_SIZE", F.size("GENDER"))
          # extract target
          .withColumn("sociodemo_targets_age", F.when(F.size("LABEL_AGE")== 1, F.col("LABEL_AGE").getItem(0)).otherwise("unknown"))
          .withColumn("sociodemo_targets_gender", F.when(F.size("GENDER") == 1, F.col("GENDER").getItem(0)).otherwise("unknown"))
         )
    # join to feature store records
    return (df_fs
            .join(df, how="left", on="user_id")
            .select("user_id", "timestamp", "sociodemo_targets_age", "sociodemo_targets_gender")
           )
    
df_final = aggregate_targets(df_process_age_categories, df_fs)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage2",
    "category": "sociodemo_targets",
    "features": {
        "sociodemo_targets_age": {
            "description": "Sociodemo target: Age.",
            "fillna_with": None,
        },
        "sociodemo_targets_gender": {
            "description": "Sociodemo target: Gender.",
            "fillna_with": None,
        },
    },
}
