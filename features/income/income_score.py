# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Income score
# MAGIC
# MAGIC This main notebook runs individual notebooks that calculate interest, URL and other (location, web behaviour) scores for the final income model. The scores are then combined, standardized and percentiles are calculated for each user and each income category. Results are written into the feature store.
# MAGIC
# MAGIC Three income categories are defined: Low (0-25k CZK/mon.), Mid (25k-45k CZK/mon.), High (45k+ CZK/mon.).

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Imports

# COMMAND ----------


import pyspark.sql.functions as F
from pyspark.sql.window import Window

from src.utils.helper_functions_defined_by_user._abcde_utils import standardize_column_positive
from src.utils.helper_functions_defined_by_user.yaml_functions import get_value_from_yaml
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Config

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The weights are designed empirically for a function that combines the partial *interest*, *URL* and *other* scores into one value using the linear combination of these scores; may be edited if needed. 
# MAGIC
# MAGIC Config:
# MAGIC * `INTERESTS_WEIGHT`: set to a lower value, since interest affinities do not scale with the number of sites visited and interest definitions may not be so reliable. 
# MAGIC * `URLS_WEIGHT`: set to a higher value, they scale with the number of sites visited and are chosen analytically from surveys.
# MAGIC * `OTHER_WEIGHT`: set to a middle value, some web behaviour features may be very reliable and act as good predictors, geolocation data not so much.

# COMMAND ----------

INTERESTS_WEIGHT = 3 / 12
URLS_WEIGHT = 5 / 12
OTHER_WEIGHT = 4 / 12

INCOME_MODELS_SUFFIXES = ["low", "mid", "high"]

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Initialize widgets and user entity

# COMMAND ----------

dbutils.widgets.dropdown("target_name", "<no target>", ["<no target>"], "01. target name")
dbutils.widgets.text("timestamp", "2020-12-12", "02. timestamp")
dbutils.widgets.dropdown("sample_data", "complete", ["complete", "sample"], "03. sample data")

# COMMAND ----------

widget_target_name = dbutils.widgets.get("target_name")
widget_timestamp = dbutils.widgets.get("timestamp")
widget_sample_data = dbutils.widgets.get("sample_data")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Run partial notebooks

# COMMAND ----------


def run_income_model_notebooks(timestamp):

    dbutils.notebook.run(
        "interest_score_income",
        10000,
        {
            "timestamp": timestamp,
        },
    )

    dbutils.notebook.run(
        "url_score_income",
        10000,
        {
            "timestamp": timestamp,
        },
    )

    dbutils.notebook.run(
        "other_score_income",
        10000,
        {
            "timestamp": timestamp,
        },
    )

run_income_model_notebooks(widget_timestamp)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Join data and standardize scores
# MAGIC Join the data from partial notebooks and calculate combined education score. The scores are then standardized and percentiles for each user are calculated.
# MAGIC
# MAGIC The loaded dataset contains final interest, URL and other scores for each defined education category.

# COMMAND ----------

def join_all_tables(interest_final, url_final, other_final, logger):
    logger.info(f"Interest score row count: {interest_final.count()}.")
    logger.info(f"URL score row count: {url_final.count()}.")
    logger.info(f"Other score row count: {other_final.count()}.")

    df = (
        interest_final.join(url_final, on=["user_id", "timestamp"], how="left")
        .join(other_final, on=["user_id", "timestamp"], how="left")
        .na.drop()
    )

    logger.info(f"Share of full rows: {100*df.count()/interest_final.count()}%.")
    return df

df_income_interest_scores = spark.read.format("delta").load(get_value_from_yaml("paths", "income_table_paths", "income_interest_scores"))
df_income_url_scores = spark.read.format("delta").load(get_value_from_yaml("paths", "income_table_paths", "income_url_scores"))
df_income_other_scores = spark.read.format("delta").load(get_value_from_yaml("paths", "income_table_paths", "income_other_scores"))

df_join_all_tables = join_all_tables(df_income_interest_scores, df_income_url_scores, df_income_other_scores, root_logger)


# COMMAND ----------

def calculate_income_scores(df):

    return df.select(
        "user_id",
        "timestamp",
        *[
            (
                INTERESTS_WEIGHT * F.col(f"final_interest_score_{model}")
                + URLS_WEIGHT * F.col(f"final_url_score_{model}")
                + OTHER_WEIGHT * F.col(f"final_other_score_{model}")
            ).alias(f"income_score_nonstd_{model}")
            for model in INCOME_MODELS_SUFFIXES
        ],
    )

df_calculate_income_scores = calculate_income_scores(df_join_all_tables)

# COMMAND ----------

def standardize_income_score(df):
    return df.select(
        "user_id",
        "timestamp",
        *[
            (standardize_column_positive(f"income_score_nonstd_{model}")).alias(
                f"income_model_score_{model}"
            )
            for model in INCOME_MODELS_SUFFIXES
        ],
    ).select(
        "user_id",
        "timestamp",
        *[F.col(f"income_model_score_{model}") for model in INCOME_MODELS_SUFFIXES],
        *[
            (
                F.percent_rank().over(Window.orderBy(f"income_model_score_{model}"))
            ).alias(f"income_model_perc_{model}")
            for model in INCOME_MODELS_SUFFIXES
        ],
    )

df_standardize_income_score = standardize_income_score(df_calculate_income_scores)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Define income model features

# COMMAND ----------

def get_features(sufixes, table_name, category_name):
    
    features_dict = {
        "table":  f"{table_name}",
        "category": f"{category_name}",
        "features":{}
        }
    
    for model in sufixes:
        features_dict['features'][f"income_model_score_{model}"] = {
        "description": f"Income model score: {model.capitalize()} income users",
        "fillna_with": None,
        "type": "numerical"
        }

    for model in sufixes:
        features_dict['features'][f"income_model_perc_{model}"] = {
        "description": f"Income model percentile: {model.capitalize()} income users",
        "fillna_with": -1.0,
        "type": "numerical"
        }

    return features_dict

metadata = get_features(INCOME_MODELS_SUFFIXES, "user", "income_score_features" )

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Write features

# COMMAND ----------


def features_income_model(df, features_name):

    return df.select(
        get_value_from_yaml("featurestorebundle", "entities", "user_entity", "id_column"),
        get_value_from_yaml("featurestorebundle", "entity_time_column"),
        *features_name,
    )

df_features_income_model = features_income_model(df_standardize_income_score, list(metadata["features"].keys()))
