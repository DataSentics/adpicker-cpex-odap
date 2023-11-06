# Databricks notebook source
# MAGIC %md
# MAGIC # Pageview features - last blog visit
# MAGIC This notebook creates the following features from `sdm_pageview` table, all in chosen last *n*-day window:
# MAGIC - web_analytics_blog_last_visit_date

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Imports & config

# COMMAND ----------

import re
import pyspark.sql.functions as F

from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Load table & fetch config values

# COMMAND ----------

df_sdm_pageview =  spark.read.format("delta").load(
    get_value_from_yaml("paths", "sdm_pageview")
)

time_window_str = get_value_from_yaml("featurestorebundle", "time_windows")[0]
time_window_int = int(re.search(r'\d+', time_window_str).group())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filter table

# COMMAND ----------

df_pageview_filtered = df_sdm_pageview.filter(F.col("page_screen_view_date") >= F.current_date() - time_window_int)

# COMMAND ----------

def calculate_days_since_blog_visit(df):
    df_days_column = df.withColumn(
        "days_since_blog_view",
        F.when(
            F.col("full_url").contains("blog"),
            F.datediff(F.current_date(), F.col("page_screen_view_date")),
        ).otherwise(None),
    )

    df_group = df_days_column.groupby("user_id").agg(
        F.min("days_since_blog_view").alias(
            f"web_analytics_blog_days_since_last_visit_{time_window_str}"
        )
    ).withColumn("timestamp", F.lit(F.current_date()))
    return df_group

df_final = calculate_days_since_blog_visit(df_pageview_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage1",
    "category": "digital_device",
    "features": {
        "web_analytics_blog_days_since_last_visit_{time_window_str}": {
            "description": "Number of days since the last visit of the blog on the web page in last {time_window_str}.",
            "fillna_with": None,
        },
    },
}
