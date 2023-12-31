# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Income models - *other* score calculation
# MAGIC This notebook serves to calculate other scores for income models (devices/web behaviour/geolocation).
# MAGIC  Useful binary features are created and they are given an empirical score for each income model (stored in azure storage).
# MAGIC  These final *other* scores are defined as sum of these scores.
# MAGIC
# MAGIC Three income categories are defined: Low (0-25k CZK/mon.), Mid (25k-45k CZK/mon.), High (45k+ CZK/mon.).

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

import pyspark.pandas as ps
import pyspark.sql.functions as F

from src.utils.helper_functions_defined_by_user._abcde_utils import (
    convert_traits_to_location_features,
)
from src.utils.read_config import config
from src.utils.helper_functions_defined_by_user.table_writing_functions import (
    write_dataframe_to_table,
)
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger
from src.utils.helper_functions_defined_by_user.feature_fetching_functions import (
    fetch_fs_stage,
)
from src.schemas.income_schemas import get_income_other_scores

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Config

# COMMAND ----------

DEVICES = ["digital_device", "digital_general"]
INCOME_MODELS_SUFFIXES = ["low", "mid", "high"]

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Initialize widgets and user entity

# COMMAND ----------

dbutils.widgets.text("timestamp", "")

# COMMAND ----------

widget_timestamp = dbutils.widgets.get("timestamp")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Part one - web behaviour features
# MAGIC Create web behaviour features from FS digital features.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Get web behaviour features

# COMMAND ----------



web_features_list = config.web_features_list

# COMMAND ----------


def read_fs(timestamp, features_to_load):
    df = fetch_fs_stage(timestamp, stage=1, feature_list=features_to_load).withColumn(
        "timestamp", F.to_timestamp(F.col("timestamp"))
    )
    return df


df_fs = read_fs(widget_timestamp, config.web_features_list)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Convert features to flags
# MAGIC Convert some interesting features to binary flags.

# COMMAND ----------


def get_web_binary_features(df):
    return df.select(
        "user_id",
        "timestamp",
        (F.col("web_analytics_page_search_engine_most_common_7d") == "seznam")
        .cast("int")
        .alias("seznam_flag"),
        (F.col("web_analytics_device_browser_most_common_7d") == "safari")
        .cast("int")
        .alias("safari_flag"),
        (F.col("web_analytics_device_type_most_common_7d") == "desktop")
        .cast("int")
        .alias("desktop_flag"),
        (F.col("web_analytics_device_type_most_common_7d") == "TV")
        .cast("int")
        .alias("TV_flag"),
        (F.col("web_analytics_num_distinct_device_categories_7d") >= 2)
        .cast("int")
        .alias("more_devices_flag"),
    )


df_get_web_binary_features = get_web_binary_features(df_fs)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Part two - location traits
# MAGIC Create location binary flags from cpex location traits.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### Load user traits and location traits map - DEPRECATED
# MAGIC This won't be used since we don't have any location traits/segments on Piano.

# COMMAND ----------

df_location_traits_map = (
    spark.read.format("delta")
    .load(config.paths.location_traits_map)
    .withColumnRenamed("TRAIT", "segment_id")
)

# COMMAND ----------

df_user_traits = spark.read.format("delta").load(config.paths.user_segments_path)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Join traits with location names

# COMMAND ----------


def join_location(df, df_map):
    return (
        df.join(df_map, on="segment_id", how="left")
        .withColumn("Prague", F.col("Name").like("%Praha%"))
        .withColumn("Kraj", F.col("Name").like("%kraj%"))
        .withColumn("City", F.col("Name").like("%City%"))
    )


df_join_location = join_location(df_user_traits, df_location_traits_map)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Aggregate traits for each user
# MAGIC
# MAGIC Aggregate locations for user and develop location flags

# COMMAND ----------


def aggregate_users(df):
    return (
        df.groupby("USER_ID")
        .agg(
            F.collect_set(F.col("Name")).alias("locations"),
            F.max(F.col("Prague")).alias("prague_flag"),
            F.max(F.col("City")).alias("city_flag"),
            F.max(F.col("Kraj")).alias("region_flag"),
        )
        .withColumn("num_locations", F.size(F.col("LOCATIONS")))
        .fillna(False, subset=["prague_flag", "city_flag", "region_flag"])
    )


df_aggregate_users = aggregate_users(df_join_location)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Develop binary location features
# MAGIC
# MAGIC The next cell transforms location flags to location features (*regional_town*, *countryside*).
# MAGIC  The complicated logic follows the principle of how cpex geolocation traits work;
# MAGIC  if a user is assigned a city trait, they are automatically assigned the corresponding region trait as well.
# MAGIC  This leads to a bit weird definition of the location features.
# MAGIC
# MAGIC *Prague* location is not used since too many users seem to have this location (~45%)
# MAGIC  which probably happens bacause of many IP addresses acting as Prague locations.

# COMMAND ----------


def get_location(df):
    return convert_traits_to_location_features(df).select(
        "USER_ID", "locations", "num_locations", "location_col"
    )


df_get_location = get_location(df_aggregate_users)

# COMMAND ----------


def get_location_binary_features(df):
    return df.withColumn(
        "countryside", (F.col("location_col") == "countryside").cast("int")
    ).withColumn(
        "regional_town", (F.col("location_col") == "regional_town").cast("int")
    )


df_get_location_binary_features = get_location_binary_features(df_get_location)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Join web behaviour and location data

# COMMAND ----------


def join_data(df_web, df_location):
    return df_web.join(df_location, how="left", on=["user_id"]).fillna(value=0)


df_join_data = join_data(df_get_web_binary_features, df_get_location_binary_features)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Load and sum scores
# MAGIC
# MAGIC Load and sum scores for each user. The stored weights are defined empirically, may be edited if needed.

# COMMAND ----------

df_income_other_coeffs = spark.read.format("delta").load(
    config.paths.income_other_coeffs
)

# COMMAND ----------


def multiply_scores(df, df_scores):
    cols_list = [row.flag for row in df_scores.select("flag").collect()]
    df_melted = (
        ps.DataFrame(df)
        .melt(id_vars=["user_id", "timestamp"], value_vars=cols_list, var_name="flag")
        .to_spark()
    )

    return df_melted.join(df_scores, on="flag").select(
        "user_id",
        "timestamp",
        *[
            (F.col("value") * F.col(f"score_{model}")).alias(f"scaled_score_{model}")
            for model in INCOME_MODELS_SUFFIXES
        ],
    )


df_multiply_scores = multiply_scores(df_join_data, df_income_other_coeffs)

# COMMAND ----------


def calculate_final_scores(df):
    return df.groupby("user_id").agg(
        *[
            F.sum(f"scaled_score_{model}").alias(f"final_other_score_{model}")
            for model in INCOME_MODELS_SUFFIXES
        ],
        F.max("timestamp").alias("timestamp"),
    )


df_result = calculate_final_scores(df_multiply_scores)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Save the final table

# COMMAND ----------


def save_scores(df, logger):
    logger.info(f"Saving {df.count()} rows.")
    return df.withColumn("timestamp", F.to_timestamp("timestamp"))


df_save_scores = save_scores(df_result, root_logger)
schema, info = get_income_other_scores()

write_dataframe_to_table(
    df_save_scores,
    config.paths.income_other_scores,
    schema,
    "overwrite",
    root_logger,
    table_properties=info["table_properties"],
)
