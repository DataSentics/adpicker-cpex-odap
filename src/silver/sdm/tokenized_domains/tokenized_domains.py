# Databricks notebook source
# MAGIC %md
# MAGIC #### Imports

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from collections import namedtuple
from datetime import date, datetime, timedelta
from logging import Logger

from src.utils.helper_functions_defined_by_user.process_loaded_interests import (
    process_loaded_interests,
)
from src.utils.helper_functions_defined_by_user.table_writing_functions import (
    write_dataframe_to_table,
)
from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger

from schema import get_schema

Interest = namedtuple("Interest", ["keywords", "general_interest"])

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Widgets

# COMMAND ----------

dbutils.widgets.text("end_date", "", "End date")
dbutils.widgets.text("n_days", "7", "Number of days to include")
dbutils.widgets.dropdown(
    "tokens_version",
    "cleaned_unique",
    ["cleaned_unique", "stemmed_unique"],
    "Tokens version",
)
dbutils.widgets.dropdown("use_bigrams", "false", ["true", "false"], "Use bigrams")

# COMMAND ----------

widget_end_date = dbutils.widgets.get("end_date")
widget_n_days = dbutils.widgets.get("n_days")
widget_use_bigrams = dbutils.widgets.get("use_bigrams")
widget_tokens_version = dbutils.widgets.get("tokens_version")

# COMMAND ----------

# MAGIC %md #### Source

# COMMAND ----------

def load_sdm_pageview(df: DataFrame, end_date: str, n_days: str):
    # process end date
    try:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    except:
        end_date = date.today()

    # calculate start date
    start_date = end_date - timedelta(days=int(n_days))

    return df.filter(
        (F.col("page_screen_view_date") >= start_date)
        & (F.col("page_screen_view_date") < end_date)
    ).select(
        "user_id",
        "full_url",
        "URL_NORMALIZED",
        F.col("page_screen_view_date").alias("DATE"),
        "flag_publisher",
    )


# if no end date provided then current date is taken

df_silver_sdm_pageview = spark.read.format("delta").load(
    get_value_from_yaml("paths", "sdm_table_paths", "sdm_pageview")
)

df_pageview = load_sdm_pageview(df_silver_sdm_pageview, widget_end_date, widget_n_days)

# COMMAND ----------

def load_sdm_url(df: DataFrame, tokens_version, use_bigrams, logger: Logger):
    # take cleaned unique as default option
    tokens_col = "URL_TOKENS_ALL_CLEANED_UNIQUE"
    bigrams_col = "URL_TOKENS_ALL_CLEANED_UNIQUE_BIGRAMS"

    if tokens_version == "cleaned_unique":
        tokens_col = "URL_TOKENS_ALL_CLEANED_UNIQUE"
        bigrams_col = "URL_TOKENS_ALL_CLEANED_UNIQUE_BIGRAMS"
    elif tokens_version == "stemmed_unique":
        tokens_col = "URL_TOKENS_ALL_STEMMED_UNIQUE"
        bigrams_col = "URL_TOKENS_ALL_STEMMED_UNIQUE_BIGRAMS"
    else:
        logger.warning(
            f"{tokens_version} is not supported. Tokens are sourced from column: {tokens_col}"
        )

    # add bigrams
    if use_bigrams == "true":
        df = df.withColumn(tokens_col, F.concat(tokens_col, bigrams_col))

    return df.select(
        "URL_NORMALIZED",
        F.col(tokens_col).alias("TOKENS"),
    )


df_silver_sdm_url = spark.read.format("delta").load(
    get_value_from_yaml("paths", "sdm_table_paths", "sdm_url")
)

df_sdm_url = load_sdm_url(
    df_silver_sdm_url, widget_tokens_version, widget_use_bigrams, root_logger
)

# COMMAND ----------

def read_interests(df: DataFrame, tokens_version):
    loaded_interests = process_loaded_interests(
        df=df, general_interests=False, keyword_variant=tokens_version
    )

    return loaded_interests


df_interests = spark.read.format("delta").load(
    get_value_from_yaml("paths", "interests_table_paths", "interests_definition")
)

subinterests = read_interests(df_interests, widget_tokens_version)

# COMMAND ----------

# MAGIC %md Combine SDM tables

# COMMAND ----------

def url_tokenized(df_pageview: DataFrame, df_url: DataFrame):
    return df_pageview.join(df_url, on="URL_NORMALIZED", how="left")


df_url_tokenized = url_tokenized(df_pageview, df_sdm_url)

# COMMAND ----------

# MAGIC %md #### Save output

# COMMAND ----------

def create_tokenized_domains(df_url_tokenized: DataFrame, subinterests):
    interest_keywords = [interest.keywords for interest in subinterests.values()]

    vocabulary = list(set([item for sublist in interest_keywords for item in sublist]))
    vocabulary = spark.createDataFrame(vocabulary, T.StringType())

    output = df_url_tokenized.select(
        "USER_ID", "TOKENS", "URL_NORMALIZED", "DATE"
    ).withColumn("TOKEN", F.explode("TOKENS"))

    return output.join(F.broadcast(vocabulary), output.TOKEN == vocabulary.value).drop(
        "VALUE"
    )


df_tokenized_domains = create_tokenized_domains(df_url_tokenized, subinterests["tuple"])

schema, info = get_schema()

write_dataframe_to_table(
    df_tokenized_domains,
    get_value_from_yaml("paths", "sdm_table_paths", "sdm_tokenized_domains"),
    schema,
    "overwrite",
    root_logger,
    info["partition_by"],
    info["table_properties"],
)
