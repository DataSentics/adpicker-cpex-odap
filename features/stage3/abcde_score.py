# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # ABCDE segmentation
# MAGIC This notebook calculates the ABCDE segmentation scores and percentiles. Income and education scores are fetched from FS, combined and ABCDE brackets are manually defined for suitable combinations. Percentiles for each bracket, A-E are then calculated and written into FS.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

import pyspark.sql.functions as F

from pyspark.sql.window import Window

from src.utils.helper_functions_defined_by_user.yaml_functions import get_value_from_yaml

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Config
# MAGIC Income has higher impact in the original ABCDE segmentation, therefore its weight coefficient is set higher than the education weight.

# COMMAND ----------

INCOME_MODELS_SUFFIXES = ["low", "mid", "high"]
EDUCATION_MODELS_SUFFIXES = ["zs", "ss_no", "ss_yes", "vs"]
ABCDE_MODELS_PREFIXES = ["A", "B", "C", "D", "E"]
INCOME_COEFF = 1
EDUCATION_COEFF = 0.7

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Initialize widgets and user entity

# COMMAND ----------

dbutils.widgets.text("timestamp", "", "timestamp")

# COMMAND ----------

widget_timestamp = dbutils.widgets.get("timestamp")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Fetch percentiles from FS

# COMMAND ----------

def read_fs():
    perc_features = [
        f"income_model_perc_{model}" for model in INCOME_MODELS_SUFFIXES
    ] + [f"edu_model_perc_{model}" for model in EDUCATION_MODELS_SUFFIXES]
    
    fs_stage2 = (
        spark.read.table("odap_features_user.user_stage2")
        .select("user_id", "timestamp", *perc_features)
        .filter(F.col("timestamp") == widget_timestamp)
    )
    return fs_stage2

df_fs = read_fs()
df_fs.display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Define segmentation brackets
# MAGIC Combine income and eductaion scores, create ABCDE scores by defining suitable income/education combinations.

# COMMAND ----------

def calculate_combinations(df):
    """
    Function to succintly calculate scores for all possible combinations of income and education levels (as linear combination of these two scores).

    :param df: dataframe containing user_ids, timestamps and all income and education model percentile features from FS.
    :return: dataframe with scores of all income/education combinations for each user, with timestamps.
    """

    def _calc_for_level(level: str):
        return [
            (
                INCOME_COEFF * F.col(f"income_model_perc_{level}")
                + EDUCATION_COEFF * F.col(f"edu_model_perc_{model}")
            ).alias(f"{level}_{model}")
            for model in EDUCATION_MODELS_SUFFIXES
        ]

    return df.select(
        F.col("user_id"),
        F.col("timestamp"),
        *[col for level in INCOME_MODELS_SUFFIXES for col in _calc_for_level(level)],
    )

df_calculate_combinations = calculate_combinations(df_fs)

# COMMAND ----------

def define_brackets(df):
    """
    Bracket combinations are designed manually to roughly correspond to income/education combinations.
    """
    return df.select(
        "user_id",
        "timestamp",
        F.greatest("high_vs", "high_ss_yes").alias("A_score"),
        F.greatest("high_ss_yes", "mid_vs").alias("B_score"),
        F.greatest("mid_ss_yes", "mid_ss_no", "high_ss_no").alias("C_score"),
        F.greatest("low_ss_yes", "mid_zs").alias("D_score"),
        F.greatest("low_zs", "low_ss_no").alias("E_score"),
    )

df_define_brackets = define_brackets(df_calculate_combinations)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Calculate percentiles

# COMMAND ----------

def calculate_percentiles(df):
    return df.select(
        "user_id",
        "timestamp",
        *[F.col(f"{model}_score") for model in ABCDE_MODELS_PREFIXES],
        *[
            (F.percent_rank().over(Window.orderBy(f"{model}_score"))).alias(
                f"{model}_perc"
            )
            for model in ABCDE_MODELS_PREFIXES
        ],
    )

df_final = calculate_percentiles(df_define_brackets)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Metadata

# COMMAND ----------

metadata =  {
    "table":  "user_stage3",
    "category": "ABCDE_score_features",
    "features": {
        "{category}_score": {
            "description": "ABCDE model score: Segment {category}",
            "fillna_with": None,
        },
        "{category}_perc": {
            "description": "ABCDE model percentile: Segment {category}",
            "fillna_with": -1.0,
        },
    }
    }
