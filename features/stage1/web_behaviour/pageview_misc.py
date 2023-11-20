# Databricks notebook source
# MAGIC %md
# MAGIC # Pageview features - miscellaneous
# MAGIC This notebook creates the following features from `sdm_pageview` table, all in chosen last *n*-day window:
# MAGIC - web_analytics_pageviews_sum
# MAGIC - web_analytics_page_search_engine_most_common
# MAGIC - web_analytics_page_search_engine_last_used
# MAGIC - owner_names

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


def page_screen_with_search_engine(df):
    return df.withColumn(
        "search_engine",
        F.when(F.col("search_engine") == "", "")
        .when(F.col("search_engine").contains("seznam"), "seznam")
        .when(F.col("search_engine").contains("google"), "google")
        .when(F.col("search_engine").contains("centrum"), "centrum")
        .when(F.col("search_engine").contains("bing"), "bing")
        .when(F.col("search_engine").contains("yahoo"), "yahoo")
        .otherwise(None),
    )  # replacement of other categories by NULL


df_pageview_with_search_engine = page_screen_with_search_engine(df_pageview_filtered)

# COMMAND ----------


def calculate_pageview_misc(df):
    window_spec = Window.partitionBy("user_id").orderBy(
        F.col("page_screen_view_timestamp").desc()
    )

    df_windowed = df.withColumn(
        "search_engine_last_used", F.max("search_engine").over(window_spec)
    )

    df_grouped = (
        df_windowed.groupby("user_id")
        .agg(
            F.mode("search_engine").alias(
                f"web_analytics_page_search_engine_most_common_{time_window_str}"
            ),
            F.first("search_engine_last_used").alias(
                f"web_analytics_page_search_engine_last_used_{time_window_str}"
            ),
            F.count(F.lit(1)).alias(f"web_analytics_pageviews_sum_{time_window_str}"),
            F.collect_set("owner_name").alias("owner_name_set"),
        )
        .withColumn(
            f"owner_names_{time_window_str}", F.concat_ws(",", F.col("owner_name_set"))
        )
        .drop("owner_name_set")
    ).withColumn("timestamp", F.lit(F.current_date()))
    return df_grouped


df_final = calculate_pageview_misc(df_pageview_with_search_engine)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage1",
    "category": "digital_device",
    "features": {
        "web_analytics_pageviews_sum_{time_window_str}": {
            "description": "Total number of viewed pageviews on a bank website in last {time_window_str}.",
            "fillna_with": 0,
        },
        "web_analytics_page_search_engine_most_common{time_window_str}": {
            "description": "Most common page screen search engine used by client in last {time_window_str}.",
            "fillna_with": None,
        },
        "web_analytics_page_search_engine_last_used_{time_window_str}": {
            "description": "Last page screen search engine used by client in last {time_window_str}.",
            "fillna_with": None,
        },
        "owner_names_{time_window_str}": {
            "description": "List of publishers in last {time_window_str}.",
            "fillna_with": "unknown",
        },
    },
}
