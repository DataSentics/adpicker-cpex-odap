# Databricks notebook source
import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

from src.schemas.sdm_schemas import get_schema_sdm_pageview
from src.utils.helper_functions_defined_by_user.table_writing_functions import (
    write_dataframe_to_table,
)
from src.utils.read_config import config
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md #### Load preprocessed

# COMMAND ----------

df_silver_sdm_preprocessed = spark.read.format("delta").load(
    config.paths.sdm_preprocessed
)

# COMMAND ----------


def add_pageview_ids_and_empty_cols(df: DataFrame):
    return (
        df.withColumn("PAGE_VIEWS", F.explode("PAGES_TIMES"))
        .withColumn("URL", F.col("page_views").getItem(0))
        .filter(~(F.col("URL") == ""))
        .withColumn("TIMESTAMP", F.to_timestamp(F.col("page_views").getItem(1)))
        .withColumn("URL_NORMALIZED", F.col("page_views").getItem(2))
        .withColumn("OWNER_ID", F.col("page_views").getItem(3))
        .withColumn("OWNER_NAME", F.col("page_views").getItem(4))
        .withColumn("SEARCH_ENGINE", F.lower(F.col("page_views").getItem(5)))
        .withColumn("FLAG_ADVERTISER", F.col("page_views").getItem(6).cast("boolean"))
        .withColumn("FLAG_PUBLISHER", F.col("page_views").getItem(7).cast("boolean"))
        .selectExpr("*", "uuid() AS PAGEVIEW_ID")
    )


df_added_pageview_ids_and_empty_cols = add_pageview_ids_and_empty_cols(
    df_silver_sdm_preprocessed
)

# COMMAND ----------

# MAGIC %md #### Append

# COMMAND ----------


def save_pageview_table(df: DataFrame):
    return df.select(
        F.col("PAGEVIEW_ID").alias("page_screen_view_id"),
        F.lit(None).cast("string").alias("page_screen_title"),
        F.col("TIMESTAMP").alias("page_screen_view_timestamp"),
        F.col("DATE").alias("page_screen_view_date"),
        F.col("URL_NORMALIZED"),
        F.col("URL").alias("full_url"),
        F.lit(None).cast("string").alias("hostname"),
        F.lit(None).cast("string").alias("page_path"),
        F.lit(None).cast("string").alias("page_path_level_1"),
        F.lit(None).cast("string").alias("page_path_level_2"),
        F.lit(None).cast("string").alias("page_path_level_3"),
        F.lit(None).cast("string").alias("page_path_level_4"),
        "search_engine",
        "flag_advertiser",
        "flag_publisher",
        "session_id",
        "user_id",
        "owner_name",
    )


df_pageview_table = save_pageview_table(df_added_pageview_ids_and_empty_cols)

schema_sdm_pageview, info_sdm_pageview = get_schema_sdm_pageview()

write_dataframe_to_table(
    df_pageview_table,
    config.paths.sdm_pageview,
    schema_sdm_pageview,
    "append",
    root_logger,
    info_sdm_pageview["partition_by"],
    info_sdm_pageview["table_properties"],
)
