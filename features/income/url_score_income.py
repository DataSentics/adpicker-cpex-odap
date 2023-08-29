# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Income models - URL score calculation
# MAGIC
# MAGIC This notebook serves to calculate URL scores for income models. The most common URLs are given a score based on marketing surveys (stored in azure storage). These scores are then joined to users' pageview and these values are averaged for each user, scaled by total number of sites visited. We transform and standardize those values to produce final URL scores for each income bracket. 
# MAGIC
# MAGIC Three income categories are defined: Low (0-25k CZK/mon.), Mid (25k-45k CZK/mon.), High (45k+ CZK/mon.).

# COMMAND ----------

# MAGIC %run ../../../app/bootstrap

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

import daipe as dp
import numpy as np
import pyspark.sql.functions as F

from adpickercpex.solutions._abcde_utils import (
    standardize_column_sigmoid,
    calculate_count_coefficient,
)

from datetime import date, timedelta, datetime
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from scipy.stats import boxcox
from logging import Logger

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

# MAGIC %md 
# MAGIC
# MAGIC ## Load pageview data
# MAGIC Load pageview data for the last *n* days, (*n*=7 is the default).

# COMMAND ----------

@dp.notebook_function()
def create_text_widgets(widgets: dp.Widgets):
    widgets.add_text("timestamp", "", "timestamp")
    widgets.add_text("n_days", "7", "Number of days to include")

# COMMAND ----------

# Current date taken if not explicitly specified
@dp.transformation(
    dp.read_table("silver.sdm_pageview"),
    dp.get_widget_value("timestamp"),
    dp.get_widget_value("n_days"),
    display=False,
)
def load_sdm_pageview(df: DataFrame, end_date: str, n_days: str, logger: Logger):
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

# COMMAND ----------

@dp.transformation(load_sdm_pageview)
def check_row_number(df, logger: Logger):
    logger.info(f"Loaded {df.count()} pageviews.")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Load URL data

# COMMAND ----------

@dp.transformation(dp.read_table("silver.sdm_url"))
def load_sdm_url(df: DataFrame):
    return df.select(
        "URL_NORMALIZED", "URL_TITLE", "URL_DOMAIN_1_LEVEL", "URL_DOMAIN_2_LEVEL"
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Join data
# MAGIC Join the pageviews with URL data.

# COMMAND ----------

@dp.transformation(load_sdm_pageview, load_sdm_url)
def join_datasets(df1, df2):
    return df1.join(df2, on="URL_NORMALIZED", how="left")

# COMMAND ----------

@dp.transformation(join_datasets)
def check_row_number_joined(df, logger: Logger):
    logger.info(f"Number of rows after join: {df.count()}.")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Join data with empirical scores 
# MAGIC Load empirical scores from azure storage and join them with the pageviews data.

# COMMAND ----------

@dp.transformation(dp.read_table("silver.income_url_coeffs"))
def load_url_scores(df):
    return df.withColumnRenamed("domain", "URL_DOMAIN_2_LEVEL")

# COMMAND ----------

@dp.transformation(join_datasets, load_url_scores)
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

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Calculate average of URL scores
# MAGIC Calculate average of URL scores for each user; also caluclate their total number of sites visited and list of URL for monitoring purposes.

# COMMAND ----------

@dp.transformation(join_with_scores)
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

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Convert URL counts
# MAGIC Convert URL counts to scaling coefficient using custom function

# COMMAND ----------

@dp.transformation(collect_urls_for_users)
def calculate_scaling_coefficient(df):
    return df.withColumn(
        "count_coefficient", calculate_count_coefficient(F.col("url_count"))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Multiply URL scores 
# MAGIC Multiply URL scores by count scaling coefficient.

# COMMAND ----------

@dp.transformation(calculate_scaling_coefficient)
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

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Transform and standardize scores
# MAGIC
# MAGIC Use Box-Cox transform to improve distribution of the score and then standardize using `StandardScaler`.
# MAGIC
# MAGIC The values for Box-Cox transformation are shifted to be positive via a static constant (0.01); the magnitude of the shift should not matter ([see link](https://stats.stackexchange.com/questions/399435/which-constant-to-add-when-applying-box-cox-transformation-to-negative-values)).

# COMMAND ----------

@dp.transformation(calculate_nonstd_url_scores)
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

# COMMAND ----------

@dp.transformation(box_cox_transform)
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

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Scale scores
# MAGIC Finally use a sigmoid function to squash the URL scores into a (-1, 1) interval.

# COMMAND ----------

@dp.transformation(standard_scaler)
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

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Save final table

# COMMAND ----------

@dp.transformation(url_score_final)
@dp.table_overwrite("silver.income_url_scores")
def save_scores(df, logger: Logger):
    logger.info(f"Saving {df.count()} rows.")
    return df
