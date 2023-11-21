# Databricks notebook source
# MAGIC %md
# MAGIC # Pageview features - web visits
# MAGIC This notebook creates the following features from `sdm_pageview` table, all in chosen last *n*-day window:
# MAGIC - web_analytics_total_visits

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

df_sdm_pageview = spark.read.format("delta").load(
    config.paths.sdm_pageview
)

time_window_str = config.featurestorebundle.time_windows[0]
time_window_int = int(re.search(r"\d+", time_window_str).group())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filter table

# COMMAND ----------

df_pageview_filtered = df_sdm_pageview.filter(
    F.col("page_screen_view_date") >= F.current_date() - time_window_int
)

# COMMAND ----------


def calculate_distinct_web_visits(df):
    df_grouped = (
        df.filter(~F.col("full_url").contains("blog"))
        .groupby("user_id")
        .agg(
            F.count_distinct("page_screen_view_timestamp").alias(
                f"web_analytics_total_visits_{time_window_str}"
            )
        )
    ).withColumn("timestamp", F.lit(F.current_date()))
    return df_grouped


df_final = calculate_distinct_web_visits(df_pageview_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage1",
    "category": "digital_device",
    "features": {
        "web_analytics_total_visits_{time_window_str}": {
            "description": "Number of total web page visits by the client in last {time_window_str}.",
            "fillna_with": 0,
        },
    },
}
