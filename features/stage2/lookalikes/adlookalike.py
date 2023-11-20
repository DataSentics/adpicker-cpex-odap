# Databricks notebook source
# MAGIC %md #### Imports

# COMMAND ----------

import json
import mlflow
import pyspark.sql.functions as F

from logging import Logger
from pyspark.sql import Window
from pyspark.sql.dataframe import DataFrame

from src.utils.helper_functions_defined_by_user.logger import instantiate_logger
from src.utils.helper_functions_defined_by_user._functions_ml import ith
from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)
from src.utils.helper_functions_defined_by_user.feature_fetching_functions import (
    fetch_fs_stage,
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Config

# COMMAND ----------

LOOKALIKE_PROBABILITY_PREFIX = "lookalike_prob_"
LOOKALIKE_PERCENTILE_PREFIX = "lookalike_perc_"
LOOKALIKE_FEATURE_PREFIX = "lookalike_target_"

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md Widgets

# COMMAND ----------

dbutils.widgets.text("timestamp", "")

# COMMAND ----------

timestamp = dbutils.widgets.get("timestamp")

# COMMAND ----------

# MAGIC %md #### Load Feature Store

# COMMAND ----------


def read_fs(timestamp):
    df = fetch_fs_stage(timestamp, stage=1).withColumn(
        "timestamp", F.to_timestamp(F.col("timestamp"))
    )
    df = df.withColumn("owner_names", F.split(F.col("owner_names_7d"), ","))
    return df


df_fs = read_fs(timestamp)

# COMMAND ----------

# MAGIC %md #### Load models from database

# COMMAND ----------


def load_lookalikes_to_score():
    lookalike_path = get_value_from_yaml("paths", "lookalike_path")
    df_lookalike = spark.read.format("delta").load(lookalike_path)

    return df_lookalike


df_loaded_lookalikes = load_lookalikes_to_score()

# COMMAND ----------

# MAGIC %md #### Score lookalikes

# COMMAND ----------


def score_lookalikes(df: DataFrame, lookalike_models_df: DataFrame, logger: Logger):
    fs_columns_to_drop = [
        colname for colname in df.columns if colname not in ["user_id", "timestamp"]
    ]
    lookalike_models_df = lookalike_models_df.toPandas()

    for _, row in lookalike_models_df.iterrows():
        model = row["Model"]

        percentile_col_name = f"{LOOKALIKE_PERCENTILE_PREFIX}{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}"
        probability_col_name = f"{LOOKALIKE_PROBABILITY_PREFIX}{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}"
        logger.info(
            f"percentile_col_name: {percentile_col_name}, probability_col_name: {probability_col_name}"
        )

        if model is None:  # model hasn't been trained yet
            df = df.withColumn(percentile_col_name, F.lit(-1)).withColumn(
                probability_col_name, F.lit(-1)
            )
        else:
            model_info = json.loads(model)
            stage = "None"
            model_registry_uri = model_info["mlf_model"] + stage
            model_obj = mlflow.spark.load_model(model_registry_uri)

            # drop columns created by pipeline to avoid duplicates
            original_cols = set(df.columns)
            df = model_obj.transform(df)
            pipeline_cols = set(df.columns)
            cols_to_drop = list(pipeline_cols - original_cols)

            df = (
                df.withColumn(probability_col_name, ith("probability", F.lit(1)))
                .withColumn(
                    percentile_col_name,
                    F.percent_rank().over(
                        Window.partitionBy().orderBy(probability_col_name)
                    ),
                )
                .drop(*cols_to_drop)
            )

    return df.drop(*fs_columns_to_drop)


df_final = score_lookalikes(df_fs, df_loaded_lookalikes, root_logger)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage2",
    "category": "lookalike_features",
    "features": {
        "lookalike_prob_{model}": {
            "description": "Lookalike probability for model {model}.",
            "fillna_with": None,
        },
        "lookalike_perc_{model}": {
            "description": "Lookalike percentile for model {model}.",
            "fillna_with": -1.0,
        },
    },
}
