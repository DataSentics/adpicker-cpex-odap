# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Sociodemo models application
# MAGIC This notebook serves to apply the trained sociodemo ML models to the data and write the probabilities/percentiles features into the FS.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import mlflow
import pyspark.sql.functions as F
import os

from logging import Logger
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
)
from pyspark.sql.window import Window

from src.utils.helper_functions_defined_by_user.logger import instantiate_logger
from src.utils.helper_functions_defined_by_user._functions_ml import ith
from src.utils.read_config import config
from src.utils.helper_functions_defined_by_user.feature_fetching_functions import (
    fetch_fs_stage,
)

# pylint: disable=redefined-outer-name
# pylint: disable=eval-used

# COMMAND ----------

# MAGIC %md #### Config

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

PREFIX_LIST = ["men", "women", *[f"cat{_index}" for _index in range(7)]]

SEARCH_ENGINE_VALUES_TO_REPLACE = ["ask", "maxask", "volny", "poprask"]
DEVICE_OS_VALUES_TO_REPLACE = ["ipados", "ios"]
DEVICE_TYPE_VALUES_TO_REPLACE = ["TV"]

ALLOWED_VALUES = {
    "web_analytics_device_type_most_common_7d": ["mobile", "desktop"],
    "web_analytics_device_os_most_common_7d": [
        "android",
        "windows",
        "ios",
        "macos",
        "linux",
    ],
    "web_analytics_device_browser_most_common_7d": [
        "chrome",
        "safari",
        "edge",
        "mozilla",
    ],
    "web_analytics_page_search_engine_most_common_7d": ["google", "seznam", "centrum"],
}

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Initialization

# COMMAND ----------

dbutils.widgets.text("timestamp", "")

# COMMAND ----------

timestamp = dbutils.widgets.get("timestamp")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Get features & data
# MAGIC Feature names and data from FS is fetched; the collected URLs for each user are then joined.

# COMMAND ----------


def read_fs(timestamp):
    df = fetch_fs_stage(timestamp, stage=1).withColumn(
        "timestamp", F.to_timestamp(F.col("timestamp"))
    )
    return df


df_fs = read_fs(timestamp)

# COMMAND ----------


def join_url_scores(df_fs: DataFrame, url_path: str, logger: Logger):
    df_urls = spark.read.format("delta").load(url_path)
    df_joined = df_fs.join(
        df_urls.select("user_id", "timestamp", "collected_urls"),
        on=["user_id", "timestamp"],
        how="left",
    )

    logger.info(
        f"Count of users with no URLs is {df_joined.select(F.sum(F.col('collected_urls').isNull().cast('integer'))).collect()[0][0]}."
    )
    return df_joined


url_path = config.paths.income_url_scores
df_joined = join_url_scores(df_fs, url_path, root_logger)

# COMMAND ----------


def get_schemas(models_dict: dict):
    with open(
        f'../features/stage2/sociodemo/schemas/socdemo_gender_schema_{models_dict["gender_male"].split("/")[-3]}.txt',
        "r",
        encoding="utf-8",
    ) as f:
        schema_gender = eval(f.readlines()[0])

    with open(
        f'../features/stage2/sociodemo/schemas/socdemo_age_schema_{models_dict["ageGroup_0_17"].split("/")[-3]}.txt',
        "r",
        encoding="utf-8",
    ) as f:
        schema_age = eval(f.readlines()[0])

    schema_gender.extend(schema_age)
    schema_both = list(set(schema_gender))
    schema_both.remove(StructField("label", StringType(), True))

    return schema_both


# COMMAND ----------

models_dict = config.paths.models.sociodemo.os.getenv("APP_ENV")
schemas_both = get_schemas(models_dict)

# COMMAND ----------


def get_cols_from_schema(schema):
    unique_urls = []
    columns_list = ["collected_urls"]
    num_cols = []
    cat_cols = []

    for f in schema:
        if f.name.split("_")[-1] == "flag":
            unique_urls.append((".").join(f.name.split("_")[:-1]))
            num_cols.append(f.name)

        else:
            columns_list.append(f.name)
            if isinstance(f.dataType, StringType):
                cat_cols.append(f.name)
            elif isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType)):
                num_cols.append(f.name)
            else:
                raise BaseException(f"{f.name} is unknown type {f.dataType}.")
    return unique_urls, columns_list, num_cols, cat_cols


# COMMAND ----------

unique_urls, columns_list, num_cols, cat_cols = get_cols_from_schema(schemas_both)
columns_list.extend(["user_id", "timestamp"])

# COMMAND ----------


def choose_features(df: DataFrame, columns_list):
    df = df.select(
        *columns_list,
        *[
            (F.array_contains(F.col("collected_urls"), domain).cast("int")).alias(
                f"{domain.replace('.', '_')}_flag"
            )
            for domain in unique_urls
        ],
    ).drop("collected_urls")
    return df


df_fs_features = choose_features(df_joined, columns_list)

# COMMAND ----------


def replace_rare_values(df: DataFrame, num_cols, cat_cols):
    for key, value in ALLOWED_VALUES.items():
        df = df.withColumn(
            key, F.when(F.col(key).isin(value), F.col(key)).otherwise("None")
        )

    df = df.fillna(0, subset=num_cols)
    df = df.fillna("None", subset=cat_cols)
    return df


df_fs_replaced = replace_rare_values(df_fs_features, num_cols, cat_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Apply sociodemo models

# COMMAND ----------


def apply_models(df: DataFrame, models_dict: dict, logger: Logger):
    for model in models_dict:
        logger.info(f"Applying sociodemo model {model}, with path {models_dict[model]}")

        try:
            model_obj = mlflow.spark.load_model(models_dict[model])
            df = model_obj.transform(df)

            # drop columns created by the ML Pipeline
            to_drop_col = (
                [c for c in df.columns if c.endswith("_indexed")]
                + [c for c in df.columns if c.endswith("_encoded")]
                + [
                    "rawPrediction",
                    "probability",
                    "prediction",
                    "features",
                    "num_features",
                    "num_features_scaled",
                ]
            )
            if "num_feat_raw" in df.columns:
                to_drop_col.append("num_feat_raw")
            if "num_feat_norm" in df.columns:
                to_drop_col.append("num_feat_raw")

            df = (
                df.withColumn("score2", ith("probability", F.lit(1)))
                .withColumn(
                    "score_rank2",
                    F.percent_rank().over(Window.partitionBy().orderBy("score2")),
                )
                .withColumnRenamed("score_rank2", f"sociodemo_perc_{model}")
                .withColumnRenamed("score2", f"sociodemo_prob_{model}")
                .drop(*to_drop_col)
            )
        except BaseException as e:
            logger.error(f"ERROR: application of model: {model}, {e}")
            # null prediction
            df = df.withColumn(f"sociodemo_perc_{model}", F.lit(None).cast("double"))
            df = df.withColumn(f"sociodemo_prob_{model}", F.lit(None).cast("double"))

    # define values for male model as a complement to the female model
    df = df.withColumn(
        "sociodemo_perc_gender_female", 1 - F.col("sociodemo_perc_gender_male")
    )
    df = df.withColumn(
        "sociodemo_prob_gender_female", 1 - F.col("sociodemo_prob_gender_male")
    )

    return df.drop(*num_cols, *cat_cols)


df_final = apply_models(df_fs_replaced, models_dict, root_logger)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC #### Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage2",
    "category": "sociodemo_features",
    "features": {
        "sociodemo_perc_{model}": {
            "description": "Sociodemo percentile for model: {model}.",
            "fillna_with": -1.0,
        },
        "sociodemo_prob_{model}": {
            "description": "Sociodemo probability for model: {model}.",
            "fillna_with": None,
        },
    },
}
