# Databricks notebook source
# MAGIC %md
# MAGIC # Session features - device
# MAGIC This notebook creates the following features from `sdm_session` table, all in chosen last *n*-day window:
# MAGIC - web_analytics_num_distinct_device_categories
# MAGIC - web_analytics_channel_device_count_distinct

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Imports & config

# COMMAND ----------

import re
import pyspark.sql.functions as F

from src.utils.read_config import config

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Load table & fetch config values

# COMMAND ----------

df_sdm_session = spark.read.format("delta").load(
    config.paths.sdm_session
)

time_window_str = config.featurestorebundle.time_windows[0]
time_window_int = int(re.search(r"\d+", time_window_str).group())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filter table

# COMMAND ----------

df_session_filtered = df_sdm_session.filter(
    F.col("session_date") >= F.current_date() - time_window_int
)

# COMMAND ----------


def calculate_device_features(df):
    distinct_columns = [
        "os_name",
        "device_category",
        "device_brand_name",
    ]

    df_grouped = (
        df.na.fill("unknown", subset=distinct_columns)
        .groupby("user_id")
        .agg(
            F.count_distinct("device_category").alias(
                f"web_analytics_num_distinct_device_categories_{time_window_str}"
            ),
            F.count_distinct(*distinct_columns).alias(
                f"web_analytics_channel_device_count_distinct_{time_window_str}"
            ),
        )
        .withColumn("timestamp", F.lit(F.current_date()))
    )
    return df_grouped


df_final = calculate_device_features(df_session_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage1",
    "category": "digital_device",
    "features": {
        "web_analytics_num_distinct_device_categories_{time_window_str}": {
            "description": "Number of distinct device categories (mobile, desktop, tablet, ...) in last {time_window_str}.",
            "fillna_with": 0,
        },
        "web_analytics_channel_device_count_distinct_{time_window_str}": {
            "description": "Number of distinct devices used by client in last {time_window_str}.",
            "fillna_with": 0,
        },
    },
}
