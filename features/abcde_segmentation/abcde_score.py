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

def read_fs(feature_store):
    perc_features = [f"income_model_perc_{model}" for model in INCOME_MODELS_SUFFIXES] + [f"edu_model_perc_{model}" for model in EDUCATION_MODELS_SUFFIXES]

    return feature_store.select('user_id', 'timestamp', *perc_features).filter(F.col("timestamp") == F.lit(F.current_date()))
#this reading will be modified 
df = spark.read.format("delta").load("abfss://gold@cpexstorageblobdev.dfs.core.windows.net/feature_store/features/user_entity.delta")
df_fetch_percentiles_from_fs = read_fs(df)

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

df_calculate_ombinations = calculate_combinations(df_fetch_percentiles_from_fs)

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

df_define_brackets = define_brackets(df_calculate_ombinations)

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

df_calculate_percentiles = calculate_percentiles(df_define_brackets)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Define and write features

# COMMAND ----------

def get_features(sufixes, table_name, category_name):

    features_dict = {
    "table":  f"{table_name}",
    "category": f"{category_name}",
    "features":{}
    }

    for model in sufixes:
        features_dict['features'][f"{model}_score"] = {
        "description": f"ABCDE model score: Segment {model.upper()}",
        "fillna_with": None,
        "type": "numerical"
        }
    
    for model in sufixes:
        features_dict['features'][f"{model}_perc"] = {
        "description": f"ABCDE model percentile: Segment {model.upper()}",
        "fillna_with": -1.0,
        "type": "numerical"
        }

    return features_dict

metadata = get_features(ABCDE_MODELS_PREFIXES, "user", "abcde_score_features")

# COMMAND ----------

def features_abcde_model(df, features_name):
    
    return df.select(
        get_value_from_yaml("featurestorebundle", "entities", "user_entity", "id_column"),
        get_value_from_yaml("featurestorebundle", "entity_time_column"),
        *features_name,
    )

df_features_abcde_model = features_abcde_model(df_calculate_percentiles, list(metadata['features']))
