# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Education score
# MAGIC
# MAGIC This main notebook runs individual notebooks that calculate interest,
# MAGIC  URL and other (location, web behaviour) scores for the final education model.
# MAGIC The scores are then combined, standardized and percentiles are calculated for each user
# MAGIC  and each education category. Results are written into the feature store.
# MAGIC
# MAGIC Four education categories are defined:
# MAGIC *ZS* (zakladní škola and less),
# MAGIC *SS_no* (střední škola bez maturity),
# MAGIC *SS_yes* (střední škola s maturitou),
# MAGIC *VS* (vysoká škola).

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

import pyspark.sql.functions as F

from pyspark.sql.window import Window
from src.utils.helper_functions_defined_by_user._abcde_utils import (
    standardize_column_positive,
)
from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Config

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The weights are designed empirically for function that combines the partial *interest*,
# MAGIC  *URL* and *other* scores into one value using linear combination of these scores; may be edited if needed.
# MAGIC
# MAGIC `INTERESTS_WEIGHT` are set to a lower value,
# MAGIC  since interest affinities do not scale with number of sites visited and interest definitions may not be so reliable.
# MAGIC
# MAGIC `URLS_WEIGHT` are set to a higher value,
# MAGIC  they scale with number of sites visited and are chosen analytically from surveys.
# MAGIC
# MAGIC `OTHER_WEIGHT` are set to a middle value,
# MAGIC  some web behaviour features may be very reliable and act as good predictors, geolocation data not so much.

# COMMAND ----------

EDUCATION_MODELS_SUFFIXES = ["zs", "ss_no", "ss_yes", "vs"]
INTERESTS_WEIGHT = 3 / 12
URLS_WEIGHT = 5 / 12
OTHER_WEIGHT = 4 / 12

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Initialize widgets and user entity

# COMMAND ----------

dbutils.widgets.text("timestamp", "")

# COMMAND ----------

timestamp = dbutils.widgets.get("timestamp")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Run partial notebooks

# COMMAND ----------


def check_for_daily_run(scores_table, timestamp):
    path = get_value_from_yaml("paths", scores_table)
    df_scores = (
        spark.read.format("delta").load(path).filter(F.col("timestamp") == timestamp)
    )
    if df_scores.count() > 0:
        return True
    else:
        return False


# COMMAND ----------


def run_education_model_notebooks(timestamp):
    if check_for_daily_run("education_interest_scores", timestamp) is False:
        dbutils.notebook.run(
            "../src/solutions/education/interest_score_edu",
            10000,
            {
                "timestamp": timestamp,
            },
        )

    if check_for_daily_run("education_url_scores", timestamp) is False:
        dbutils.notebook.run(
            "../src/solutions/education/url_score_edu",
            10000,
            {
                "timestamp": timestamp,
            },
        )

    if check_for_daily_run("education_other_scores", timestamp) is False:
        dbutils.notebook.run(
            "../src/solutions/education/other_score_edu",
            10000,
            {
                "timestamp": timestamp,
            },
        )


run_education_model_notebooks(timestamp)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Join data and standardize scores
# MAGIC Join the data from partial notebooks and calculate combined education score.
# MAGIC  The scores are then standardized and percentiles for each user are calculated.
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

    logger.info(f"Share of full rows: {100 * df.count() / interest_final.count()}%.")
    return df


df_education_interest_scores = spark.read.format("delta").load(
    get_value_from_yaml("paths", "education_interest_scores")
)
df_education_url_scores = spark.read.format("delta").load(
    get_value_from_yaml("paths", "education_url_scores")
)
df_education_other_scores = spark.read.format("delta").load(
    get_value_from_yaml("paths", "education_other_scores")
)

df_join_all_tables = join_all_tables(
    df_education_interest_scores,
    df_education_url_scores,
    df_education_other_scores,
    root_logger,
)

# COMMAND ----------


def calculate_edu_scores(df):
    return df.select(
        "user_id",
        "timestamp",
        *[
            (
                INTERESTS_WEIGHT * F.col(f"final_interest_score_{model}")
                + URLS_WEIGHT * F.col(f"final_url_score_{model}")
                + OTHER_WEIGHT * F.col(f"final_other_score_{model}")
            ).alias(f"edu_score_nonstd_{model}")
            for model in EDUCATION_MODELS_SUFFIXES
        ],
    )


df_calculate_edu_scores = calculate_edu_scores(df_join_all_tables)

# COMMAND ----------


def standardize_edu_score(df):
    return df.select(
        "user_id",
        "timestamp",
        *[
            (standardize_column_positive(f"edu_score_nonstd_{model}")).alias(
                f"edu_model_score_{model}"
            )
            for model in EDUCATION_MODELS_SUFFIXES
        ],
    ).select(
        "user_id",
        "timestamp",
        *[F.col(f"edu_model_score_{model}") for model in EDUCATION_MODELS_SUFFIXES],
        *[
            (F.percent_rank().over(Window.orderBy(f"edu_model_score_{model}"))).alias(
                f"edu_model_perc_{model}"
            )
            for model in EDUCATION_MODELS_SUFFIXES
        ],
    )


df_standardize_edu_score = standardize_edu_score(df_calculate_edu_scores)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Define education model features

# COMMAND ----------


def get_features(sufixes, table_name, category_name):
    features_dict = {
        "table": f"{table_name}",
        "category": f"{category_name}",
        "features": {},
    }

    for model in sufixes:
        features_dict["features"][f"edu_model_score_{model}"] = {
            "description": f"Education model score: {model.upper()} education level",
            "fillna_with": None,
            "type": "numerical",
        }

    for model in sufixes:
        features_dict["features"][f"edu_model_perc_{model}"] = {
            "description": f"Education model percentile: {model.upper()} education level",
            "fillna_with": -1.0,
            "type": "numerical",
        }

    return features_dict


metadata = get_features(EDUCATION_MODELS_SUFFIXES, "user", "edu_score_features")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Write features

# COMMAND ----------


def features_income_model(df, features_name):
    return df.select(
        get_value_from_yaml(
            "featurestorebundle", "entities", "user_entity", "id_column"
        ),
        get_value_from_yaml("featurestorebundle", "entity_time_column"),
        *features_name,
    )


df_final = features_income_model(
    df_standardize_edu_score, list(metadata["features"].keys())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage2",
    "category": "education_score_features",
    "features": {
        "edu_model_score_{category}": {
            "description": "Education model score: {category} education level",
            "fillna_with": None,
        },
        "edu_model_perc_{category}": {
            "description": "Income model percentile: {category} education level",
            "fillna_with": -1.0,
        },
    },
}
