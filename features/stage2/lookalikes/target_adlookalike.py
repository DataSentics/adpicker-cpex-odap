# Databricks notebook source
# MAGIC %md #### Imports

# COMMAND ----------

import pyspark.sql.functions as F

from logging import Logger
from pyspark.sql.dataframe import DataFrame

from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger
from src.utils.helper_functions_defined_by_user.feature_fetching_functions import (
    fetch_fs_stage,
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Config

# COMMAND ----------

LOOKALIKE_FEATURE_PREFIX = "lookalike_target_"

dbutils.widgets.text("timestamp", "")

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

timestamp = dbutils.widgets.get("timestamp")

# COMMAND ----------

# MAGIC %md #### Load inputs

# COMMAND ----------

# MAGIC %md Load lookalike models

# COMMAND ----------


def get_defined_lookalikes():
    lookalike_path = get_value_from_yaml("paths", "lookalike_path")
    df_lookalike = spark.read.format("delta").load(lookalike_path)
    df_lookalike = (
        df_lookalike.withColumn(
            "TP_DMP_id", F.regexp_replace("TP_DMP_id", ",", "_")
        )  # replace ',' by '_' as df column names cannot contain ','
        .select("TP_DMP_id", "client_name", "TP_DMP_type")
        .distinct()
    )
    return df_lookalike


df_lookalikes = get_defined_lookalikes()

# COMMAND ----------

# MAGIC %md Load feature store

# COMMAND ----------


def read_fs(timestamp):
    df = fetch_fs_stage(timestamp, stage=1, feature_list=[]).withColumn(
        "timestamp", F.to_timestamp(F.col("timestamp"))
    )
    return df


df_fs = read_fs(timestamp)

# COMMAND ----------

# MAGIC %md Load user segments

# COMMAND ----------


def get_user_segments():
    user_segments_path = get_value_from_yaml("paths", "user_segments_path")
    df_segments = spark.read.format("delta").load(user_segments_path)

    return df_segments


df_user_segments = get_user_segments()

# COMMAND ----------

# MAGIC %md #### Assign target

# COMMAND ----------


def assign_lookalikes_target(
    df_fs: DataFrame, df_user_segments: DataFrame, df_models: DataFrame, logger: Logger
):
    df_models = df_models.toPandas()

    for _, row in df_models.iterrows():
        # extract trait
        lookalike_column_name = f"{LOOKALIKE_FEATURE_PREFIX}{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}"
        segment_id = row["TP_DMP_id"]
        segment_ids_list = segment_id.split("_")

        try:
            # filter observations with given trait
            df_feature = (
                df_user_segments.filter(F.col("segment_id").isin(segment_ids_list))
                # if there are more segments in a lookalike this effectively creates target as their union
                .withColumn(lookalike_column_name, F.lit(1))
                .select("user_id", lookalike_column_name)
                .dropDuplicates()
            )

            # join to feature store record
            df_fs = df_fs.join(df_feature, how="left", on="user_id")
        except BaseException as e:
            logger.error(f"ERROR: adding LaL target for: {segment_id}, {e}")

    return df_fs.fillna(0)


df_fs_targets = assign_lookalikes_target(
    df_fs, df_user_segments, df_lookalikes, root_logger
)

# COMMAND ----------


def get_array_of_lookalikes(df_fs_targets: DataFrame, df_lookalikes):
    feature_names = [
        colname
        for colname in df_fs_targets.columns
        if colname not in ["user_id", "timestamp"]
    ]

    for feature_name in feature_names:
        df_fs_targets = df_fs_targets.withColumn(
            f"temp_{feature_name}",
            F.when(
                F.col(feature_name) == 1,
                feature_name.replace(LOOKALIKE_FEATURE_PREFIX, ""),
            ).otherwise(None),
        )

    df_fs_targets_array = df_fs_targets.withColumn(
        "lookalike_targets", F.concat_ws(",", *[f"temp_{c}" for c in feature_names])
    ).drop(*[f"temp_{c}" for c in feature_names])

    return df_fs_targets_array


df_final = get_array_of_lookalikes(df_fs_targets, df_lookalikes)

# COMMAND ----------

# MAGIC %md #### Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage2",
    "category": "lookalike_targets",
    "features": {
        "lookalike_target_{model}": {
            "description": "Lookalike target for model {model}.",
            "fillna_with": None,
        },
        "lookalike_targets": {
            "description": "Array of lookalikes for user.",
            "fillna_with": None,
        },
    },
}
