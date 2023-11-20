# Databricks notebook source
# MAGIC %md #### Imports

# COMMAND ----------

from collections import namedtuple
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F

from src.utils.helper_functions_defined_by_user.process_loaded_interests import (
    process_loaded_interests,
)
from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)
from src.utils.helper_functions_defined_by_user.indexes_calculator import (
    indexes_calculator,
)

Interest = namedtuple("Interest", ["keywords", "general_interest"])

# COMMAND ----------

dbutils.widgets.dropdown(
    "target_name", "<no target>", ["<no target>"], "01. target name"
)
dbutils.widgets.text("timestamp", "2020-12-12", "02. timestamp")
dbutils.widgets.dropdown(
    "sample_data", "complete", ["complete", "sample"], "03. sample data"
)
dbutils.widgets.dropdown(
    "tokens_version",
    "cleaned_unique",
    ["cleaned_unique", "stemmed_unique"],
    "04. tokens version",
)
dbutils.widgets.dropdown("use_biagrams", "false", ["true", "false"], "05. use biagrams")

# COMMAND ----------

widget_target_name = dbutils.widgets.get("target_name")
widget_timestamp = dbutils.widgets.get("timestamp")
widget_sample_data = dbutils.widgets.get("sample_data")
widget_tokens_version = dbutils.widgets.get("tokens_version")
widget_use_biagrams = dbutils.widgets.get("use_biagrams")

# COMMAND ----------

# MAGIC %md #### Load table

# COMMAND ----------


def tokenized_domains(df: DataFrame, entity, timestamp):
    return df.withColumn(entity, F.lit(timestamp))


df_sdm_tokenized_domains = spark.read.format("delta").load(
    get_value_from_yaml("paths", "sdm_tokenized_domains")
)
df_tokenized_domains = tokenized_domains(
    df_sdm_tokenized_domains, "timestamp", widget_timestamp
)

# COMMAND ----------

# MAGIC %md #### Interests

# COMMAND ----------


def read_interests(df: DataFrame, tokens_version):
    loaded_interests = process_loaded_interests(
        df=df, general_interests=True, keyword_variant=tokens_version
    )

    return loaded_interests


df_intersts_definition = spark.read.format("delta").load(
    get_value_from_yaml("paths", "interests_definition")
)

dict_interests = read_interests(df_intersts_definition, widget_tokens_version)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Add interests to domains

# COMMAND ----------


def get_tokenized_domains_interests(tokenized_impressions: DataFrame, interests: dict):
    tokenized_impressions = tokenized_impressions.drop("date")

    return tokenized_impressions.select(
        *tokenized_impressions.columns,
        *[
            F.when(F.col("token").isin(interest.keywords), 1)
            .otherwise(0)
            .alias(subinterest)
            for subinterest, interest in interests.items()
        ],
    )


df_tokenized_domains_interests = get_tokenized_domains_interests(
    df_tokenized_domains, dict_interests["tuple"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Metadata

# COMMAND ----------


def get_features(dict_with_interests: dict, table_name, category_name):
    features_dict = {
        "category": f"{category_name}",
        "table": f"{table_name}",
        "features": {},
    }

    for subinterest, interest in dict_with_interests.items():
        features_dict["features"][f"{subinterest}"] = {
            "description": f"General Interest: {interest.general_interest}; Subinterest: {subinterest.replace('_', ' ')}",
            "fillna_with": 0.0,
        }
    return features_dict


# COMMAND ----------

metadata = get_features(
    dict_interests["tuple"], "user", "general_interest_affinity_features"
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Further interest calculation
# MAGIC

# COMMAND ----------


def get_interests_stats(tokenized_domains_interests: DataFrame, interest_names: list):
    n_row = F.count(F.lit(1))

    interests_stats = (
        tokenized_domains_interests.select(*interest_names)
        .groupBy()
        .agg(*[F.log(n_row / F.sum(F.col(x))).alias(x) for x in interest_names])
    )

    interests_list = list(interests_stats.columns)

    return interests_stats.select(
        F.greatest(*interests_list).alias("max"),
        F.least(*interests_list).alias("min"),
        *(F.col(col).alias(f"stat_{col}") for col in interests_stats.columns),
    )


df_interests_stats = get_interests_stats(
    df_tokenized_domains_interests, dict_interests["names"]
)

# COMMAND ----------


def get_joined_interests_with_stats(
    tokenized_domains_interests: DataFrame, interests_stats: DataFrame
):
    return tokenized_domains_interests.join(interests_stats, how="cross")


df_joined_interests_with_stats = get_joined_interests_with_stats(
    df_tokenized_domains_interests, df_interests_stats
)

# COMMAND ----------


def features_digi_interests(
    joined_interests_with_stats: DataFrame,
    interest_names: list,
    id_column="user_id",
    time_column="timestamp",
):
    interest_affinities = indexes_calculator(
        joined_interests_with_stats,
        interest_names,
        windows_number=1,
        level_of_distinction=[id_column, time_column],
    )

    return interest_affinities.select(
        id_column,
        time_column,
        *interest_names,
    )


df_final = features_digi_interests(
    df_joined_interests_with_stats, list(metadata["features"].keys())
).withColumn("timestamp", F.current_date())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Metadata

# COMMAND ----------

metadata = {
    "table": "user_stage1",
    "category": "general_interest_affinity_features",
    "features": {
        "ad_interest_affinity_{interest}": {
            "description": "General Interest: {interest}; Subinterest: ad interest affinity {interest}",
            "fillna_with": 0.0,
        }
    },
}
