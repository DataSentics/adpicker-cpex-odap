# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Income models - URL score calculation
# MAGIC
# MAGIC This notebook serves to calculate URL scores for income models. The most common URLs are given a score based on marketing surveys (stored in azure storage). These scores are then joined to users' pageview and these values are averaged for each user, scaled by total number of sites visited. We transform and standardize those values to produce final URL scores for each income bracket. 
# MAGIC
# MAGIC Three income categories are defined: Low (0-25k CZK/mon.), Mid (25k-45k CZK/mon.), High (45k+ CZK/mon.).

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

from pyspark.sql.dataframe import DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
import pyspark.sql.functions as F

import numpy as np
from datetime import date, timedelta, datetime
from scipy.stats import boxcox
from logging import Logger

from src.utils.helper_functions_defined_by_user._abcde_utils import (
    standardize_column_sigmoid,
    calculate_count_coefficient,
)
from src.utils.helper_functions_defined_by_user.yaml_functions import get_value_from_yaml
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger
from src.utils.helper_functions_defined_by_user.table_writing_functions import write_dataframe_to_table
from src.schemas.income_schemas import get_income_url_scores
from src.utils.read_config import config


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Config
# MAGIC
# MAGIC Configure suffixes for income model brackets and sharpness of sigmoidal standardization function.

# COMMAND ----------

INCOME_MODELS_SUFFIXES = ["low", "mid", "high"]
SHARPNESS = 1

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Load pageview data
# MAGIC Load pageview data for the last *n* days, (*n*=7 is the default).

# COMMAND ----------

dbutils.widgets.text("timestamp", "", "timestamp")
dbutils.widgets.text("n_days", "7", "Number of days to include")

# COMMAND ----------

widget_timestamp = dbutils.widgets.get("timestamp")
widget_n_days = dbutils.widgets.get("n_days")

# COMMAND ----------

# Current date taken if not explicitly specified
def load_sdm_pageview(df: DataFrame, end_date: str, n_days: str, logger):
    # process end date
    try:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        logger.info(f"Date set from widget value: {end_date}")
    except:
        end_date = date.today()
        logger.info(f"No initial date set; setting as today: {end_date}")

    # calculate start date
    start_date = end_date - timedelta(days=int(n_days))

    return (
        df.filter(
            (F.col("page_screen_view_date") >= start_date)
            & (F.col("page_screen_view_date") < end_date)
        )
        .select(
            "user_id",
            "full_url",
            "URL_NORMALIZED",
            F.col("page_screen_view_date").alias("DATE"),
            "flag_publisher",
            F.lit(end_date).cast("timestamp").alias("timestamp"),
        )
    )

#df_sdm_pageview = spark.read.format("delta").load(get_value_from_yaml("paths", "sdm_pageview"))
df_sdm_pageview = spark.read.format("delta").load(config.paths.sdm_pageview)
df_load_sdm_pageview = load_sdm_pageview(df_sdm_pageview, widget_timestamp, widget_n_days, root_logger)

# COMMAND ----------

def check_row_number(df, logger):
    logger.info(f"Loaded {df.count()} pageviews.")

check_row_number(df_load_sdm_pageview, root_logger)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Load URL data

# COMMAND ----------

def load_sdm_url(df: DataFrame):
    return df.select(
        "URL_NORMALIZED", "URL_TITLE", "URL_DOMAIN_1_LEVEL", "URL_DOMAIN_2_LEVEL"
    )

df_sdm_url = spark.read.format("delta").load(config.paths.sdm_url)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Join data
# MAGIC Join the pageviews with URL data.

# COMMAND ----------

def join_datasets(df1, df2):
    return df1.join(df2, on="URL_NORMALIZED", how="left")

df_join_datasets = join_datasets(df_load_sdm_pageview, df_sdm_url)

# COMMAND ----------

def check_row_number_joined(df, logger):
    logger.info(f"Number of rows after join: {df.count()}.")

check_row_number_joined(df_join_datasets, root_logger)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Join data with empirical scores 
# MAGIC Load empirical scores from azure storage and join them with the pageviews data.

# COMMAND ----------

def load_url_scores():
    df = spark.read.format("delta").load(config.paths.income_url_coeffs) 
    return df.withColumnRenamed("domain", "URL_DOMAIN_2_LEVEL")

df_income_url_coeffs = load_url_scores()

# COMMAND ----------

def join_with_scores(df1, df2):
    return (
        df1.join(df2, on="URL_DOMAIN_2_LEVEL", how="left")
        .fillna(0, subset=[f"score_{model}" for model in INCOME_MODELS_SUFFIXES])
        .select(
            "user_id",
            "timestamp",
            "URL_DOMAIN_2_LEVEL",
            "URL_NORMALIZED",
            *[f"score_{model}" for model in INCOME_MODELS_SUFFIXES],
        )
    )

df_join_with_scores = join_with_scores(df_join_datasets, df_income_url_coeffs)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Calculate average of URL scores
# MAGIC Calculate average of URL scores for each user; also caluclate their total number of sites visited and list of URL for monitoring purposes.

# COMMAND ----------

def collect_urls_for_users(df):
    return df.groupby("user_id").agg(
        *[
            F.mean(f"score_{model}").alias(f"score_average_{model}")
            for model in INCOME_MODELS_SUFFIXES
        ],
        F.count("timestamp").alias("url_count"),
        F.collect_list("URL_DOMAIN_2_LEVEL").alias("collected_urls"),
        F.max("timestamp").alias("timestamp"),
    )

df_collect_urls_for_users = collect_urls_for_users(df_join_with_scores)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Convert URL counts
# MAGIC Convert URL counts to scaling coefficient using custom function

# COMMAND ----------

def calculate_scaling_coefficient(df):
    return df.withColumn(
        "count_coefficient", calculate_count_coefficient(F.col("url_count"))
    )

df_calculate_scaling_coefficient = calculate_scaling_coefficient(df_collect_urls_for_users)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Multiply URL scores 
# MAGIC Multiply URL scores by count scaling coefficient.

# COMMAND ----------

def calculate_nonstd_url_scores(df):
    return df.select(
        "user_id",
        "timestamp",
        "collected_urls",
        *[
            (F.col("count_coefficient") * F.col(f"score_average_{model}")).alias(
                f"url_score_nonstd_{model}"
            )
            for model in INCOME_MODELS_SUFFIXES
        ],
    )

df_calculate_nonstd_url_scores = calculate_nonstd_url_scores(df_calculate_scaling_coefficient)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Transform and standardize scores
# MAGIC
# MAGIC Use Box-Cox transform to improve distribution of the score and then standardize using `StandardScaler`.
# MAGIC
# MAGIC The values for Box-Cox transformation are shifted to be positive via a static constant (0.01); the magnitude of the shift should not matter ([see link](https://stats.stackexchange.com/questions/399435/which-constant-to-add-when-applying-box-cox-transformation-to-negative-values)).

# COMMAND ----------

def box_cox_transform(df):
    nonstd_columns = [f"url_score_nonstd_{model}" for model in INCOME_MODELS_SUFFIXES]
    
    df_pandas = df.select(
        "user_id", "timestamp", "collected_urls", *nonstd_columns
    ).toPandas()
    
    for col in nonstd_columns:        
        scores = np.array(df_pandas[col]).reshape(-1)
        scores = (
            scores + np.abs(np.min(scores)) + 0.01
        )  # all values need to be positive for Box-Cox transformation
        fitted = boxcox(scores)
        df_pandas[col] = fitted[0]

    df_spark = spark.createDataFrame(df_pandas)
    return df_spark

df_box_cox_transform = box_cox_transform(df_calculate_nonstd_url_scores)

# COMMAND ----------

def standard_scaler(df):
    nonstd_columns = ["url_score_nonstd_" + model for model in INCOME_MODELS_SUFFIXES]
    std_columns = ["url_score_std_" + model for model in INCOME_MODELS_SUFFIXES]

    vec_ass = VectorAssembler(
        inputCols=nonstd_columns, outputCol="features", handleInvalid="skip"
    )
    df_vec = vec_ass.transform(df)

    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features", withMean=True
    )
    df_scaled = scaler.fit(df_vec).transform(df_vec)

    return df_scaled.withColumn(
        "array", vector_to_array(F.col("scaled_features"))
    ).select(
        "user_id",
        "timestamp",
        "collected_urls",
        *[(F.col("array")[i]).alias(col) for i, col in enumerate(std_columns)]
    )

df_standard_scaler = standard_scaler(df_box_cox_transform)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Scale scores
# MAGIC Finally use a sigmoid function to squash the URL scores into a (-1, 1) interval.

# COMMAND ----------

def url_score_final(df):
    return df.select(
        "user_id",
        "timestamp",
        "collected_urls",
        *[
            standardize_column_sigmoid(f"url_score_std_{model}", SHARPNESS).alias(
                f"final_url_score_{model}"
            )
            for model in INCOME_MODELS_SUFFIXES
        ],
    )

df_result = url_score_final(df_standard_scaler)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Save final table

# COMMAND ----------

def save_scores(df, logger):
    logger.info(f"Saving {df.count()} rows.")
    return df

df_save_scores = save_scores(df_result, root_logger)
schema, info = get_income_url_scores() 

write_dataframe_to_table(
    df_save_scores,
    config.paths.income_url_scores,
    schema,
    "overwrite",
    root_logger,
    table_properties=info["table_properties"],)
