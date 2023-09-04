# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Income models - *other* score calculation
# MAGIC This notebook serves to calculate other scores for income models (devices/web behaviour/geolocation). Useful binary features are created and they are given an empirical score for each income model (stored in azure storage). These final *other* scores are defined as sum of these scores. 
# MAGIC
# MAGIC Three income categories are defined: Low (0-25k CZK/mon.), Mid (25k-45k CZK/mon.), High (45k+ CZK/mon.).

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

import pyspark.pandas as ps
import pyspark.sql.functions as F

from src.utils.helper_functions_defined_by_user._abcde_utils import convert_traits_to_location_features
from src.utils.helper_functions_defined_by_user.yaml_functions import get_value_from_yaml
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger
from schemas import get_income_other_scores

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Config

# COMMAND ----------

DEVICES = ["digital_device", "digital_general"]
INCOME_MODELS_SUFFIXES = ["low", "mid", "high"]

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Initialize widgets and user entity

# COMMAND ----------

dbutils.widgets.text("timestamp", "", "timestamp")
dbutils.widgets.text("n_days", "7", "Number of days to include")

# COMMAND ----------

widget_timestamp = dbutils.widgets.get("timestamp")
widget_n_days = dbutils.widgets.get("n_days")

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

def get_web_features_list(df):
    feat_list = df.filter(F.col("category").isin(DEVICES))
    return [element.feature for element in feat_list.collect()]

df_metadata = spark.read.format("delta").load(get_value_from_yaml("paths", "feature_store_paths", "metadata"))
list_web_features_list = get_web_features_list(df_metadata)
print(list_web_features_list)

# COMMAND ----------

def read_fs(feature_store, list_features):

    return feature_store.select('user_id', 'timestamp', *list_features).filter(F.col("timestamp") == F.lit(F.current_date()))
#this reading will be modified 
df = spark.read.format("delta").load(get_value_from_yaml("paths", "feature_store_paths", "user_entity_fs"))
df_read_web_features_from_fs = read_fs(df, list_web_features_list) 

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

df_get_web_binary_features= get_web_binary_features(df_read_web_features_from_fs)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Part two - location traits
# MAGIC Create location binary flags from cpex location traits.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ### Load user traits and location traits map

# COMMAND ----------


df_location_traits_map = spark.read.format("delta").load(get_value_from_yaml("paths", "location_table_paths", "location_traits_map"))

# COMMAND ----------

df_user_traits = spark.read.format("delta").load(get_value_from_yaml("paths", "user_table_paths", "user_traits"))

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Join traits with location names

# COMMAND ----------

def join_location(df, df_map):
    return (
        df.join(df_map, on="TRAIT", how="left")
        .withColumn("Prague", F.col("Name").like("%Praha%"))
        .withColumn("Kraj", F.col("Name").like("%kraj%"))
        .withColumn("City", F.col("Name").like("%City%"))
    )

df_join_location = join_location(df_user_traits, df_user_traits)

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
            F.max(F.to_timestamp(F.col("END_DATE"))).alias("timestamp"),
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
# MAGIC The next cell transforms location flags to location features (*regional_town*, *countryside*). The complicated logic follows the principle of how cpex geolocation traits work; if a user is assigned a city trait, they are automatically assigned the corresponding region trait as well. This leads to a bit weird definition of the location features.
# MAGIC
# MAGIC *Prague* location is not used since too many users seem to have this location (~45%) which probably happens bacause of many IP addresses acting as Prague locations.

# COMMAND ----------

def get_location(df):
    return convert_traits_to_location_features(df).select(
        "USER_ID", "timestamp", "locations", "num_locations", "location_col"
    )

df_get_location = get_location(df_aggregate_users)

# COMMAND ----------

def get_location_binary_features(df):
    return df.withColumn(
        "countryside", (F.col("location_col") == "countryside").cast("int")
    ).withColumn(
        "regional_town", (F.col("location_col") == "regional_town").cast("int")
    )

df_get_location_bnary_features = get_location_binary_features(df_get_location)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Join web behaviour and location data

# COMMAND ----------

def join_data(df_web, df_location):
    return df_web.join(df_location, how="left", on=["user_id", "timestamp"]).fillna(
        value=0
    )

df_join_data = join_data(df_get_web_binary_features, df_get_location_bnary_features)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Load and sum scores
# MAGIC
# MAGIC Load and sum scores for each user. The stored weights are defined empirically, may be edited if needed.

# COMMAND ----------

df_income_other_coeffs = spark.read.format("delta").load(get_value_from_yaml("paths", "income_table_paths", "income_other_coeffs"))

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

df_final = calculate_final_scores(df_multiply_scores)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Save the final table

# COMMAND ----------

def save_scores(df, logger):
    logger.info(f"Saving {df.count()} rows.")
    return df

df_save_scores = save_scores(df_final, root_logger)
schema, info = get_income_other_scores()

write_dataframe_to_table(
    df_save_scores,
    get_value_from_yaml("paths", "income_table_paths", "income_other_scores"),
    schema,
    "overwrite",
    root_logger,
    table_properties=info["table_properties"],
 )
