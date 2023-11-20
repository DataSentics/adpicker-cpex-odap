# Databricks notebook source
# MAGIC %md
# MAGIC # Session features - category
# MAGIC
# MAGIC This notebook creates the following features from `sdm_session` table, all in chosen last *n*-day window:
# MAGIC - web_analytics_device_type_most_common
# MAGIC - web_analytics_device_type_last_used
# MAGIC - web_analytics_device_browser_most_common
# MAGIC - web_analytics_device_browser_last_used
# MAGIC - web_analytics_device_os_most_common
# MAGIC - web_analytics_device_os_last_used

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Imports & config

# COMMAND ----------

import re
import pyspark.sql.functions as F

from pyspark.sql.window import Window

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

# MAGIC %md
# MAGIC #### Create features
# MAGIC

# COMMAND ----------


def calculate_category_features(df):
    window_spec = Window.partitionBy("user_id").orderBy(
        F.col("session_start_datetime").desc()
    )

    df_windowed_cols = (
        df.withColumn(
            "device_category_last_used", F.max("device_category").over(window_spec)
        )
        .withColumn("browser_last_used", F.max("browser_name").over(window_spec))
        .withColumn("os_last_used", F.max("os_name").over(window_spec))
    )

    df_grouped = (
        df_windowed_cols.groupby("user_id")
        .agg(
            F.mode("device_category").alias(
                f"web_analytics_device_type_most_common_{time_window_str}"
            ),
            F.first("device_category_last_used").alias(
                f"web_analytics_device_type_last_used_{time_window_str}"
            ),
            F.mode("browser_name").alias(
                f"web_analytics_device_browser_most_common_{time_window_str}"
            ),
            F.first("browser_last_used").alias(
                f"web_analytics_device_browser_last_used_{time_window_str}"
            ),
            F.mode("os_name").alias(
                f"web_analytics_device_os_most_common_{time_window_str}"
            ),
            F.first("os_last_used").alias(
                f"web_analytics_device_os_last_used_{time_window_str}"
            ),
        )
        .withColumn("timestamp", F.lit(F.current_date()))
    )
    return df_grouped


df_final = calculate_category_features(df_session_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage1",
    "category": "digital_device",
    "features": {
        "web_analytics_device_{element}_most_common_{time_window_str}": {
            "description": "Most common device {element} used by client in last {time_window_str}.",
            "fillna_with": None,
        },
        "web_analytics_device_{element}_last_used_{time_window_str}": {
            "description": "Last device {element} used by client in the last {time_window_str}.",
            "fillna_with": None,
        },
    },
}
