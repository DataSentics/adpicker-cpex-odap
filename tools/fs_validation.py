# Databricks notebook source
import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)

# COMMAND ----------

tables_to_validate = [
    "income_url_scores",
    # "income_url_coeffs",
    "income_interest_scores",
    # "income_interest_coeffs",
    "income_other_scores",
    # "income_other_coeffs",
    "education_interest_scores",
    # "education_interest_coeffs",
    "education_url_scores",
    # "education_url_coeffs",
    "education_other_scores",
    # "education_other_coeffs"
]

# COMMAND ----------

def _get_latest_version(df_path: str) -> int:    
    df_history = spark.sql(f"DESCRIBE HISTORY delta.`{df_path}`")

    return df_history.agg({"version": "max"}).collect()[0][0]

# COMMAND ----------

def _get_columns_to_compare(df: DataFrame) -> list():
    return [col_name for col_name in df.columns if "final" in col_name]

# COMMAND ----------

def validate_data_between_two_versions(df_name: str, coef: float = 0.1) -> tuple():
    """
    Function that should be able to validate if there are any discrepancies between last two versions of calculated Feature Store rows - bigger than specific thrash-hold

    Parameters:
    df_name: Name of table that is specified in config file
    coef: Thrash-hold under which we expect that rows are calculated just fine
    """
    df_path = get_value_from_yaml("paths", df_name)
    latest_version = _get_latest_version(df_path)

    df_1 = spark.read.format("delta").option("versionAsOf", latest_version).load(df_path)
    df_2 = spark.read.format("delta").option("versionAsOf", latest_version - 1).load(df_path)

    columns_to_validate = _get_columns_to_compare(df_1)

    # get random sample of last version table
    df_1_s = df_1.sample(False, 0.1).limit(10000)

    df_join = df_1_s.alias("df_1").join(
        df_2.alias("df_2"),
        on="user_id",
        how="inner"
    ).withColumn("diff", F.lit(0))  # add column to check validity and for further filtering

    df_diff = (
        df_join
        .withColumn("diff", F.lit(0))
    )

    for col in columns_to_validate:    
        df_diff = (
            df_diff
            .withColumn("diff", F.when(F.abs((F.col(f"df_1.{col}")-F.col(f"df_2.{col}"))/F.col(f"df_1.{col}")) < F.lit(coef), F.col("diff")
                        ).otherwise(F.lit(1))
            )
        )

    df_diff = (
        df_diff
        .filter(F.col("diff") == F.lit(1))
    )

    return (True if df_diff.count() == 0 else False, df_diff)

# COMMAND ----------

for tab in tables_to_validate:
    print("------------------------")
    print(tab)
    x, y = validate_data_between_two_versions(tab)
    print(x)
    display(y)
