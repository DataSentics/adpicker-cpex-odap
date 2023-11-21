# Databricks notebook source
# MAGIC %md
# MAGIC # Session features - visit period
# MAGIC This notebook creates the following features from `sdm_session` table, all in chosen last *n*-day window:
# MAGIC - web_analytics_visit_time_most_common

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


def web_with_visit_period(df):
    return df.withColumn(
        "visit_hour", F.hour(F.col("session_start_datetime"))
    ).withColumn(
        "visit_period",
        F.when(F.col("visit_hour").between(5, 8), "Early morning")
        .when(F.col("visit_hour").between(9, 11), "Late morning")
        .when(F.col("visit_hour").between(12, 15), "Early afternoon")
        .when(F.col("visit_hour").between(16, 17), "Late afternoon")
        .when(F.col("visit_hour").between(18, 20), "Early evening")
        .when(F.col("visit_hour").between(21, 23), "Late evening")
        .otherwise("Night"),
    )


df_session_with_visit_period = web_with_visit_period(df_session_filtered)

# COMMAND ----------


def calculate_visit_time_most_common(df):
    df_visit_period = (
        df.groupby("user_id")
        .agg(
            F.mode("visit_period").alias(
                f"web_analytics_visit_time_most_common_{time_window_str}"
            )
        )
        .withColumn("timestamp", F.lit(F.current_date()))
    )
    return df_visit_period


df_final = calculate_visit_time_most_common(df_session_with_visit_period)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage1",
    "category": "digital_device",
    "features": {
        "web_analytics_visit_time_most_common_{time_window_str}": {
            "description": "Most common part of the day when visiting website in last {time_window_str}.",
            "fillna_with": None,
        },
    },
}
