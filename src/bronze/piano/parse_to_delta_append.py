# Databricks notebook source
# MAGIC %md
# MAGIC # Parse data to bronze - PIANO

# COMMAND ----------

import re
from datetime import datetime, timedelta
from functools import reduce
from logging import Logger
from math import ceil

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.dataframe import DataFrame

from src.schemas.bronze_schema import get_schema_cpex_piano_cleansed
from src.utils.helper_functions_defined_by_user.date_functions import get_max_date
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger
from src.utils.helper_functions_defined_by_user.table_writing_functions import (
    write_dataframe_to_table,
    delta_table_exists,
)
from src.utils.read_config import config

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md
# MAGIC Widgets

# COMMAND ----------

dbutils.widgets.dropdown("series_length", "long", ["short", "long"], "series length")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load raw data & parse to bronze

# COMMAND ----------

# MAGIC %md Create empty table

# COMMAND ----------

widget_series_length = dbutils.widgets.get("series_length")

# COMMAND ----------


# create empty if not exists - releavnt only for long table (as append is used)
def create_bronze_cpex_piano(series_length):
    if series_length == "long":
        if not delta_table_exists(config.paths.cpex_table_piano):
            schema, info = get_schema_cpex_piano_cleansed()
            df_empty = spark.createDataFrame([], schema)

            write_dataframe_to_table(
                df_empty,
                config.paths.cpex_table_piano,
                schema,
                "default",
                root_logger,
                info["partition_by"],
                info["table_properties"],
            )


create_bronze_cpex_piano(widget_series_length)

# COMMAND ----------

# MAGIC %md Get last processed date

# COMMAND ----------


def max_upload_date(df: DataFrame, series_length):
    if series_length == "long":
        return get_max_date(df=df, date_col="day", datetime_col="EVENT_TIME")
    return None


df_bronze_cpex_piano = spark.read.format("delta").load(
    config.paths.cpex_table_piano
)
df_max_upload_date = max_upload_date(df_bronze_cpex_piano, widget_series_length)

# COMMAND ----------

# MAGIC %md Select only relevant subfolders of raw folder

# COMMAND ----------


# select relevant subfolders
def get_relevant_folders(
    base_path, destination_max_date, series_length, n_hours_short, n_days_long
):
    paths = dbutils.fs.ls(base_path)
    max_date = max(
        datetime.strptime(
            p.name.replace("day=", "").replace("/", ""), "%Y-%m-%d"
        ).date()
        for p in paths
    )

    if series_length == "long":
        if destination_max_date is None:
            # n_days_long + 1 is sufficient
            date_limit = max_date - timedelta(days=n_days_long + 1)
        else:
            # select only new subfolders
            date_limit = destination_max_date.date()
        return [
            p.path
            for p in paths
            if datetime.strptime(
                p.name.replace("day=", "").replace("/", ""), "%Y-%m-%d"
            ).date()
            >= date_limit
        ]

    # short series
    n_days = ceil(n_hours_short / 24)
    date_limit = max_date - timedelta(days=n_days)
    return [
        p.path
        for p in paths
        if datetime.strptime(
            p.name.replace("day=", "").replace("/", ""), "%Y-%m-%d"
        ).date()
        >= date_limit
    ]


base_path = "/mnt/piano"
relevant_folders_list = get_relevant_folders(
    base_path,
    df_max_upload_date,
    widget_series_length,
    config.tables_options.series_length.n_hours_short,
    config.jobs_config.regular_optimization.keep_history_n_days,
)

# COMMAND ----------


def flatten_struct(schema, prefix=""):
    result = []
    for elem in schema:
        if isinstance(elem.dataType, T.StructType):
            result += flatten_struct(elem.dataType, prefix + elem.name + ".")
        else:
            result.append(F.col(prefix + elem.name).alias(elem.name))
    return result


# COMMAND ----------


def get_custom_parameters(column, param_of_interest):
    @F.pandas_udf("string")
    def get_custom_parameters_udf(column: pd.Series) -> pd.Series:
        def helper_func(param_string):
            for elem_dict in param_string:
                if elem_dict["group"] == param_of_interest:
                    return elem_dict["item"]
            return None

        return column.apply(helper_func)

    return get_custom_parameters_udf(column)


# COMMAND ----------

# MAGIC %md Load json to delta

# COMMAND ----------


def get_jsons(relevant_folders, logger: Logger):
    for folder in relevant_folders:
        folder_path = folder.split(":")[1]
        files_in_folder = dbutils.fs.ls(folder_path)
        for elem in files_in_folder:
            elem_path = elem.path.split(":")[1]

            # skip delta folder:
            if "json" not in elem_path:
                continue

            name = re.search(
                r".*\d{4}-\d{2}-\d{2}/(.*?_*[0-9]+)\.json", elem_path
            ).group(1)
            save_path = folder_path + "delta/" + name

            # skip if already exists (checking with os library doesn't work):
            try:
                dbutils.fs.ls(save_path)
                logger.info(f"Skipping {save_path} because it already exists.")
                continue
            except BaseException:
                pass

            owner_name = re.search(
                r".*\d{4}-\d{2}-\d{2}/(.*?)_*[0-9]+\.json", elem_path
            ).group(1)
            if not owner_name:
                owner_name = "unknown"

            df = (
                spark.read.json(elem_path)
                .withColumn("events", F.explode("events"))
                .select("events")
            )

            df = df.select(flatten_struct(df.schema))
            df = df.withColumn("EVENT_TIME", F.col("time").cast("timestamp"))
            df = df.drop("time")
            df = df.withColumn("SOURCE_FILE", F.lit(F.input_file_name()))
            df = df.withColumn("OWNER_NAME", F.lit(owner_name).cast(T.StringType()))
            df = df.withColumn("OWNER_ID", F.lit(None).cast(T.StringType()))
            df = df.withColumn("FLAG_ADVERTISER", F.lit(None).cast(T.StringType()))
            df = df.withColumn("FLAG_PUBLISHER", F.lit(None).cast(T.StringType()))

            df.write.format("delta").mode("overwrite").save(save_path)
            logger.info(f"Delta table written to {save_path}.")


get_jsons(relevant_folders_list, root_logger)

# COMMAND ----------


def load_raw_delta(
    relevant_folders, series_length, destination_max_date, n_hours_short, logger: Logger
):
    # pylint: disable=too-many-locals
    fetched_schema, _ = get_schema_cpex_piano_cleansed()
    user_params_expected_schema = (
        fetched_schema["userParameters"].simpleString().split(":", 1)[1]
    )
    external_id_expected_schema = (
        fetched_schema["externalUserIds"].simpleString().split(":", 1)[1]
    )

    df_to_union = []
    for folder in relevant_folders:
        # load from delta folder inside relevant folder:
        files_in_folder = dbutils.fs.ls(folder.split(":")[1] + "delta/")

        for elem in files_in_folder:
            elem_path = elem.path.split(":")[1]
            df = spark.read.format("delta").load(elem_path)

            # Standardize schema of array<struct> columns
            user_params_schema = df.select("userParameters").dtypes[0][1]
            external_id_schema = df.select("externalUserIds").dtypes[0][1]

            if external_id_schema == external_id_expected_schema:
                df = df.withColumn(
                    "externalUserIds",
                    F.array(
                        F.struct(
                            (F.col("externalUserIds.id")[0]).alias("id"),
                            (F.col("externalUserIds.type")[0]).alias("type"),
                        )
                    ),
                )
            else:
                raise Exception(
                    f"Schema mismatch in externalUserIds column on path {elem_path}!"
                )

            if user_params_schema == user_params_expected_schema:
                df = df.withColumn(
                    "userParameters",
                    F.array(
                        F.struct(
                            (F.col("userParameters.group")[0]).alias("group"),
                            (F.col("userParameters.item")[0]).alias("item"),
                        )
                    ),
                )
            elif user_params_schema == "array<string>":
                df = df.withColumn(
                    "userParameters",
                    F.array(
                        F.struct(
                            F.lit(None).cast("string").alias("group"),
                            F.lit(None).cast("string").alias("item"),
                        )
                    ),
                )
            else:
                raise Exception(
                    f"Schema mismatch in userParameters column on path {elem_path}!"
                )

            df_to_union.append(df)
            logger.info(f"Source file: {elem_path} appended")

    df = reduce(DataFrame.unionByName, df_to_union)
    logger.info(f"Number of partitions: {df.rdd.getNumPartitions()}")
    logger.info("Reduce operation on appended dataframes finished")

    df = (
        df.withColumnRenamed("url", "rp_pageurl")
        .withColumn("day", F.to_date("EVENT_TIME"))
        .withColumn("hour", F.hour(F.col("EVENT_TIME")))
        .withColumnRenamed("userId", "DEVICE")
        .withColumn("rp_pagetitle", F.lit(None).cast("string"))
        .withColumn("rp_pagekeywords", F.lit(None).cast("string"))
        .withColumn("rp_pagedescription", F.lit(None).cast("string"))
        .withColumn("rp_c_p_pageclass", F.lit(None).cast("string"))
        .withColumn("rp_c_p_publishedDateTime", F.lit(None).cast("string"))
        .withColumn("activeTime", F.col("activeTime").cast(T.IntegerType()))
        .withColumn("browserTimezone", F.col("browserTimezone").cast(T.ShortType()))
        .withColumn("colorDepth", F.col("colorDepth").cast(T.ShortType()))
        .withColumn(
            "AGE",
            get_custom_parameters(F.col("customParameters"), param_of_interest="ub"),
        )
        .withColumn(
            "GENDER",
            get_custom_parameters(F.col("customParameters"), param_of_interest="us"),
        )
    )
    if series_length == "short":
        # short series only n hours of history
        min_date = df.agg(
            F.max("EVENT_TIME") - F.expr(f"INTERVAL {n_hours_short} HOURS")
        ).collect()[0][0]
        return df.filter(F.col("EVENT_TIME") >= min_date)

    if destination_max_date:
        df = df.filter(
            F.col("day") >= F.to_date(F.lit(destination_max_date.date()))
        ).filter(F.col("EVENT_TIME") > destination_max_date)
        logger.info("Filtering by date and timestamp finished")

    return df


df_raw_data = load_raw_delta(
    relevant_folders_list,
    widget_series_length,
    df_max_upload_date,
    config.tables_options.series_length.n_hours_short,
    root_logger,
)

# COMMAND ----------

# MAGIC %md Save table

# COMMAND ----------


def save(df: DataFrame, series_length):
    schema, info = get_schema_cpex_piano_cleansed()

    if series_length == "long":
        write_dataframe_to_table(
            df,
            config.paths.cpex_table_piano,
            schema,
            "append",
            root_logger,
            info["partition_by"],
            info["table_properties"],
        )

    if series_length == "short":
        write_dataframe_to_table(
            df,
            config.paths.cpex_table_short_piano,
            schema,
            "overwrite",
            root_logger,
            info["partition_by"],
            info["table_properties"],
        )


save(df_raw_data, widget_series_length)

# COMMAND ----------


# delete all files after using them to save storage
def delete_jsons(relevant_folders, logger: Logger):
    for folder in relevant_folders:
        folder_path = folder.split(":")[1] + "delta/"
        dbutils.fs.rm(folder_path, recurse=True)
        logger.info(f"Folder {folder_path} deleted with all tables.")


delete_jsons(relevant_folders_list, root_logger)
