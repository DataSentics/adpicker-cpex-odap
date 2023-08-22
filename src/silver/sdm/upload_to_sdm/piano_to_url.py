# Databricks notebook source
import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame
from logging import Logger, getLogger
from pyspark.ml.feature import NGram
from datetime import timedelta

from src.utils.helper_functions_defined_by_user._stop_words import unwanted_tokens
from src.utils.helper_functions_defined_by_user.table_writing_functions import (
    write_dataframe_to_table,
    delta_table_exists,
)
from src.utils.helper_functions_defined_by_user.date_functions import get_max_date
from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)
from src.utils.helper_functions_defined_by_user._functions_nlp import (
    df_stemming,
    df_url_normalization,
    df_url_to_adform_format,
    df_url_to_domain,
)

from schemas import get_schema_sdm_url

# COMMAND ----------

col_to_concat = [
    "URL_NORMALIZED_KEYWORDS",
    "URL_TITLE",
    "URL_KEYWORDS",
    "URL_DESCRIPTIONS",
]
col_to_concat_cleaned = [f"{col}_CLEANED" for col in col_to_concat]
col_to_concat_stemmed = [f"{col}_STEMMED" for col in col_to_concat]

# COMMAND ----------

# MAGIC %md Create empty tables

# COMMAND ----------

def create_url_table():
    if not delta_table_exists(
        get_value_from_yaml("paths", "sdm_table_paths", "sdm_url")
    ):

        schema, info = get_schema_sdm_url()
        df_empty = spark.createDataFrame([], schema)

        write_dataframe_to_table(
            df_empty,
            get_value_from_yaml("paths", "sdm_table_paths", "sdm_url"),
            schema,
            "default",
            info["partition_by"],
            info["table_properties"],
        )


create_url_table()

# COMMAND ----------

# MAGIC %md Load cleansed table

# COMMAND ----------

df_silver_sdm_url = spark.read.format("delta").load(
    get_value_from_yaml("paths", "sdm_table_paths", "sdm_url")
)

# COMMAND ----------

def read_cleansed_data(df_silver: DataFrame, df_url: DataFrame, logger: Logger):
    # get max date in table

    destination_max_date = get_max_date(
        df=df_url, date_col="DATE_ADDED", datetime_col="TIME_ADDED", last_n_days=20
    )
    logger.info(f"maximal date in silver.sdm_url before append: {destination_max_date}")

    if destination_max_date is not None:
        # save date to filter silver table (due to potential different between 'day' and date extracted from EVENT_TIME)
        destination_max_date_safe = destination_max_date.date() - timedelta(days=2)
        df_silver = df_silver.filter(F.col("day") >= destination_max_date_safe).filter(
            F.col("EVENT_TIME") > destination_max_date
        )

    return df_silver.filter(F.col("rp_pageurl").isNotNull()).select(
        "rp_pageurl",
        "rp_pagetitle",
        "rp_pagekeywords",
        "rp_pagedescription",
        "rp_c_p_pageclass",
        "rp_c_p_publishedDateTime",
        F.to_date("EVENT_TIME").alias("DATE_ADDED"),
        F.col("EVENT_TIME").alias("TIME_ADDED"),
    )


df_bronze_cpex_piano = spark.read.format("delta").load(
    get_value_from_yaml("paths", "piano_table_paths", "cpex_table_piano")
)
root_logger = getLogger()

df_cleansed_data = read_cleansed_data(
    df_bronze_cpex_piano, df_silver_sdm_url, root_logger
)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Process URLs

# COMMAND ----------

# MAGIC %md
# MAGIC Select only new records

# COMMAND ----------

def normalize_url_data(df: DataFrame):
    return df_url_normalization(
        df.withColumn("URL_NORMALIZED", F.col("rp_pageurl")), "URL_NORMALIZED"
    )


df_normalize_url_data = normalize_url_data(df_cleansed_data)

# COMMAND ----------

# select only new URL_NORMALIZED and keep one row per URL_NORMALIZED
def get_unique_url(df_cleansed: DataFrame, df_url: DataFrame):
    return (
        df_cleansed.join(df_url, how="left_anti", on="URL_NORMALIZED")
        .groupBy("URL_NORMALIZED")
        .agg(
            F.count(F.lit(1)).alias("count"),
            F.first("rp_pageurl", ignorenulls=True).alias("rp_pageurl"),
            F.first("rp_pagetitle", ignorenulls=True).alias("rp_pagetitle"),
            F.first("rp_pagekeywords", ignorenulls=True).alias("rp_pagekeywords"),
            F.first("rp_pagedescription", ignorenulls=True).alias("rp_pagedescription"),
            F.first("rp_c_p_pageclass", ignorenulls=True).alias("rp_c_p_pageclass"),
            F.first("rp_c_p_publishedDateTime", ignorenulls=True).alias(
                "rp_c_p_publishedDateTime"
            ),
            F.min("DATE_ADDED").alias("DATE_ADDED"),
            F.min("TIME_ADDED").alias("TIME_ADDED"),
        )
    )


df_unique_url = get_unique_url(df_normalize_url_data, df_silver_sdm_url)

# COMMAND ----------

def check_for_duplicates(df, logger: Logger):
    logger.info(f"Found {df.count()} new URLs to append.")
    num_unique = (
        df.groupby("URL_NORMALIZED").count().select(F.max("count")).collect()[0][0]
    )
    if num_unique == 1:
        logger.info("No duplicates detected.")
    else:
        logger.info("Some duplicates created/no data found! Investigate please!")


df_check_for_duplicates = check_for_duplicates(df_unique_url, root_logger)

# COMMAND ----------

# MAGIC %md URL processing

# COMMAND ----------

def extract_url_formats(df: DataFrame):
    # GET URL WITH DIFFERENT DOMAIN LEVELS
    df = df_url_to_domain(
        df.withColumn("URL_DOMAIN_FULL", F.col("rp_pageurl")),
        "URL_DOMAIN_FULL",
        subdomains=0,
    )
    df = df_url_to_domain(
        df.withColumn("URL_DOMAIN_1_LEVEL", F.col("rp_pageurl")),
        "URL_DOMAIN_1_LEVEL",
        subdomains=1,
    )
    df = df_url_to_domain(
        df.withColumn("URL_DOMAIN_2_LEVEL", F.col("rp_pageurl")),
        "URL_DOMAIN_2_LEVEL",
        subdomains=2,
    )
    # ADFROM FORMAT
    return df_url_to_adform_format(
        df.withColumn("URL_ADFORM_FORMAT", F.col("rp_pageurl")), "URL_ADFORM_FORMAT"
    )


df_extract_url_formats = extract_url_formats(df_unique_url)

# COMMAND ----------

def extract_page_info(df: DataFrame):
    return (
        df.withColumnRenamed("rp_c_p_pageclass", "PAGE_TYPE")
        .withColumnRenamed("rp_c_p_publishedDateTime", "PUBLISHING_DATE")
        .withColumnRenamed("rp_pagetitle", "URL_TITLE")
    )


df_extract_page_info = extract_page_info(df_extract_url_formats)

# COMMAND ----------

def extract_url_title(df: DataFrame):
    return df_stemming(
        df=df,
        input_col="URL_TITLE",
        cleaned_col="URL_TITLE_CLEANED",
        stemmed_col="URL_TITLE_STEMMED",
        client_name="cz",
        stop_words=unwanted_tokens,
    )


df_extract_url_title = extract_url_title(df_extract_page_info)

# COMMAND ----------

def extract_url_keywords(df: DataFrame):
    df = df.withColumnRenamed("rp_pagekeywords", "URL_KEYWORDS")

    return df_stemming(
        df=df,
        input_col="URL_KEYWORDS",
        cleaned_col="URL_KEYWORDS_CLEANED",
        stemmed_col="URL_KEYWORDS_STEMMED",
        client_name="cz",
        stop_words=unwanted_tokens,
    )


df_extract_url_keywords = extract_url_keywords(df_extract_url_title)

# COMMAND ----------

def extract_url_description(df: DataFrame):
    df = df.withColumnRenamed("rp_pagedescription", "URL_DESCRIPTIONS")

    return df_stemming(
        df=df,
        input_col="URL_DESCRIPTIONS",
        cleaned_col="URL_DESCRIPTIONS_CLEANED",
        stemmed_col="URL_DESCRIPTIONS_STEMMED",
        client_name="cz",
        stop_words=unwanted_tokens,
    )


df_extract_url_description = extract_url_description(df_extract_url_keywords)

# COMMAND ----------

def extracting_url_keywords(df: DataFrame):
    return df_stemming(
        df=df,
        input_col="URL_NORMALIZED",
        cleaned_col="URL_NORMALIZED_KEYWORDS_CLEANED",
        stemmed_col="URL_NORMALIZED_KEYWORDS_STEMMED",
        client_name="cz",
        stop_words=unwanted_tokens,
    )


df_extracting_url_keywords = extracting_url_keywords(df_extract_url_description)

# COMMAND ----------

def get_url_tokens_all(df_augmented: DataFrame):
    # collecting tokens
    for col in col_to_concat_cleaned + col_to_concat_stemmed:
        df_augmented = df_augmented.withColumn(col, F.coalesce(F.col(col), F.array()))

    return (
        df_augmented.withColumn(
            "URL_TOKENS_ALL_CLEANED", F.concat(*col_to_concat_cleaned)
        )
        .withColumn("URL_TOKENS_ALL_STEMMED", F.concat(*col_to_concat_stemmed))
        .withColumn(
            "URL_TOKENS_ALL_CLEANED_UNIQUE", F.array_distinct("URL_TOKENS_ALL_CLEANED")
        )
        .withColumn(
            "URL_TOKENS_ALL_STEMMED_UNIQUE", F.array_distinct("URL_TOKENS_ALL_STEMMED")
        )
        .drop("rp_pageurl")
    )


df_get_url_tokens_all = get_url_tokens_all(df_extracting_url_keywords)

# COMMAND ----------

def get_bigrams(df: DataFrame, suffix="BIGRAMS"):
    # cols with tokens
    col_to_concat_cleaned_ngrams = [f"{col}_{suffix}" for col in col_to_concat_cleaned]
    col_to_concat_stemmed_ngrams = [f"{col}_{suffix}" for col in col_to_concat_stemmed]

    for col in col_to_concat_cleaned + col_to_concat_stemmed:
        ngram = NGram(n=2, inputCol=col, outputCol=f"{col}_{suffix}")
        df = ngram.transform(df)

    return (
        df.withColumn(
            f"URL_TOKENS_ALL_CLEANED_{suffix}", F.concat(*col_to_concat_cleaned_ngrams)
        )
        .withColumn(
            f"URL_TOKENS_ALL_STEMMED_{suffix}", F.concat(*col_to_concat_stemmed_ngrams)
        )
        .withColumn(
            f"URL_TOKENS_ALL_CLEANED_UNIQUE_{suffix}",
            F.array_distinct(f"URL_TOKENS_ALL_CLEANED_{suffix}"),
        )
        .withColumn(
            f"URL_TOKENS_ALL_STEMMED_UNIQUE_{suffix}",
            F.array_distinct(f"URL_TOKENS_ALL_STEMMED_{suffix}"),
        )
    )


df_get_bigrams = get_bigrams(df_get_url_tokens_all)

# COMMAND ----------

# MAGIC %md ### Combine the results with current URL table

# COMMAND ----------

def save_url_table(df: DataFrame):
    return df.select(
        "URL_TITLE",
        "URL_KEYWORDS",
        "URL_DESCRIPTIONS",
        "PAGE_TYPE",
        "PUBLISHING_DATE",
        "URL_DOMAIN_FULL",
        "URL_DOMAIN_1_LEVEL",
        "URL_DOMAIN_2_LEVEL",
        "URL_NORMALIZED",
        "URL_ADFORM_FORMAT",
        "URL_TITLE_CLEANED",
        "URL_TITLE_STEMMED",
        "URL_KEYWORDS_CLEANED",
        "URL_KEYWORDS_STEMMED",
        "URL_DESCRIPTIONS_CLEANED",
        "URL_DESCRIPTIONS_STEMMED",
        "URL_NORMALIZED_KEYWORDS_CLEANED",
        "URL_NORMALIZED_KEYWORDS_STEMMED",
        "URL_TOKENS_ALL_CLEANED",
        "URL_TOKENS_ALL_STEMMED",
        "URL_TOKENS_ALL_CLEANED_UNIQUE",
        "URL_TOKENS_ALL_STEMMED_UNIQUE",
        "URL_TITLE_CLEANED_BIGRAMS",
        "URL_TITLE_STEMMED_BIGRAMS",
        "URL_KEYWORDS_CLEANED_BIGRAMS",
        "URL_KEYWORDS_STEMMED_BIGRAMS",
        "URL_DESCRIPTIONS_CLEANED_BIGRAMS",
        "URL_DESCRIPTIONS_STEMMED_BIGRAMS",
        "URL_NORMALIZED_KEYWORDS_CLEANED_BIGRAMS",
        "URL_NORMALIZED_KEYWORDS_STEMMED_BIGRAMS",
        "URL_TOKENS_ALL_CLEANED_BIGRAMS",
        "URL_TOKENS_ALL_STEMMED_BIGRAMS",
        "URL_TOKENS_ALL_CLEANED_UNIQUE_BIGRAMS",
        "URL_TOKENS_ALL_STEMMED_UNIQUE_BIGRAMS",
        "TIME_ADDED",
        "DATE_ADDED",
    )


df_save_url_table = save_url_table(df_get_bigrams)

schema_sdm_url, info_sdm_url = get_schema_sdm_url()

df_save_url_table.printSchema()

write_dataframe_to_table(
    df_save_url_table,
    get_value_from_yaml("paths", "sdm_table_paths", "sdm_url"),
    schema_sdm_url,
    "append",
    info_sdm_url["partition_by"],
    info_sdm_url["table_properties"],
)
