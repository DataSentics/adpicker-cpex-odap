# Databricks notebook source
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F

from src.utils.helper_functions_defined_by_user.table_writing_functions import (
    write_dataframe_to_table,
    delta_table_exists,
)
from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)

from src.schemas.sdm_schemas import get_schema_sdm_session
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger

# COMMAND ----------

# MAGIC %md #### Load preprocessed

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

df_preprocessed = spark.read.format("delta").load(
    get_value_from_yaml("paths", "sdm_preprocessed")
)

# COMMAND ----------

# MAGIC %md #### Select relevant fields

# COMMAND ----------


def get_relevant_fields(df: DataFrame):
    return df.withColumn("flag_active_session", F.lit(True)).select(
        "session_id",
        F.col("SESSION_START_TIME").alias("session_start_datetime"),
        F.col("DATE").alias("session_date"),
        F.col("SESSION_END_TIME").alias("session_end_datetime"),
        F.col("flag_active_session"),
        "user_id",
        "browser_id",
        "device_id",
        "os_id",
        F.lit(None).cast("long").alias("geo_id"),
        F.lit(None).cast("long").alias("traffic_source_id"),
    )


df_relevant_fields = get_relevant_fields(df_preprocessed)

# COMMAND ----------

# MAGIC %md #### Append to pageview table

# COMMAND ----------


def session_table(
    df_session: DataFrame,
    df_browser: DataFrame,
    df_device: DataFrame,
    df_os: DataFrame,
):
    return (
        df_session.join(F.broadcast(df_browser), "browser_id", "left")
        .join(F.broadcast(df_device), "device_id", "left")
        .join(F.broadcast(df_os), "os_id", "left")
        .select(
            "session_id",
            "session_start_datetime",
            "session_date",
            "session_end_datetime",
            "flag_active_session",
            "user_id",
            "browser_name",
            "device_category",
            "device_full_specification",
            "device_brand_name",
            "device_marketing_name",
            F.lit(None).cast("string").alias("device_model_name"),
            F.lit(None).cast("string").alias("city"),
            F.lit(None).cast("string").alias("continent"),
            F.lit(None).cast("string").alias("subcontinent"),
            F.lit(None).cast("string").alias("country"),
            F.lit(None).cast("string").alias("region"),
            F.lit(None).cast("string").alias("metro"),
            "os_name",
            "os_version",
            F.lit(None).cast("string").alias("traffic_source_campaign"),
            F.lit(None).cast("string").alias("traffic_source_medium"),
            F.lit(None).cast("string").alias("traffic_source_source"),
            F.lit(None).cast("long").alias("traffic_source_id"),
            "os_id",
            "geo_id",
            "device_id",
            "browser_id",
        )
    )


df_silver_sdm_browser = spark.read.format("delta").load(
    get_value_from_yaml("paths", "sdm_browser")
)
df_silver_sdm_device = spark.read.format("delta").load(
    get_value_from_yaml("paths", "sdm_device")
)
df_silver_sdm_os = spark.read.format("delta").load(
    get_value_from_yaml("paths", "sdm_os")
)

df_session_table = session_table(
    df_relevant_fields, df_silver_sdm_browser, df_silver_sdm_device, df_silver_sdm_os
)

# we get the schema and infos of dataframe from which we write in table
schema_sdm_session, info_sdm_session = get_schema_sdm_session()
df_session_table.printSchema()
write_dataframe_to_table(
    df_session_table,
    get_value_from_yaml("paths", "sdm_session"),
    schema_sdm_session,
    "append",
    root_logger,
    info_sdm_session["partition_by"],
    info_sdm_session["table_properties"],
)
