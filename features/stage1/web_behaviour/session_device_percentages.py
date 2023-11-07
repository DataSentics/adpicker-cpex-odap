# Databricks notebook source
# MAGIC %md
# MAGIC # Session features - device percentages
# MAGIC This notebook creates the following features from `sdm_session` table, all in chosen last *n*-day window:
# MAGIC - web_analytics_mobile_user
# MAGIC - web_analytics_desktop_user
# MAGIC - web_analytics_tablet_user
# MAGIC - web_analytics_smart_tv_user

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Imports & config

# COMMAND ----------

import re

from functools import reduce
from operator import add

import pyspark.sql.functions as F

from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Load table & fetch config values

# COMMAND ----------

df_sdm_session = spark.read.format("delta").load(
    get_value_from_yaml("paths", "sdm_session")
)

time_window_str = get_value_from_yaml("featurestorebundle", "time_windows")[0]
time_window_int = int(re.search(r"\d+", time_window_str).group())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filter table

# COMMAND ----------

df_session_filtered = df_sdm_session.filter(
    F.col("session_date") >= F.current_date() - time_window_int
)

# COMMAND ----------


def calculate_device_percentage(df):
    device_list = ["mobile", "desktop", "smart_tv", "tablet"]

    df_grouped = df.groupby("user_id").agg(
        *[
            (F.count(F.when(F.col("device_category") == device, 1))).alias(
                f"count_{device}_{time_window_str}"
            )
            for device in device_list
        ]
    )
    df_summed = df_grouped.withColumn(
        "total",
        reduce(
            add, [F.col(f"count_{device}_{time_window_str}") for device in device_list]
        ),
    )
    df_percentages = df_summed
    for device in device_list:
        df_percentages = df_percentages.withColumn(
            f"web_analytics_{device}_user_{time_window_str}",
            F.round(F.col(f"count_{device}_{time_window_str}") / F.col("total"), 1),
        )
    cols_to_drop = [
        "total",
        *[f"count_{device}_{time_window_str}" for device in device_list],
    ]
    df_final = df_percentages.drop(*cols_to_drop).withColumn(
        "timestamp", F.lit(F.current_date())
    )
    return df_final


df_final = calculate_device_percentage(df_session_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage1",
    "category": "digital_device",
    "features": {
        "web_analytics_{device}_user_{time_window_str}": {
            "description": "Percentage of {device} visits in last {time_window_str}.",
            "fillna_with": 0,
        },
    },
}
