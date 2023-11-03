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
    # "education_other_coeffs",

    "user_entity_fs",
]

# COMMAND ----------

special_tables_to_validate = [
    "odap_features_user.latest",
    "odap_features_user.user",
    "odap_features_user.user_stage1",
    "odap_features_user.user_stage2",
    "odap_features_user.user_stage3",
]

# COMMAND ----------

def _get_latest_version(df_path: str) -> int:
    df_history = spark.sql(f"DESCRIBE HISTORY delta.`{df_path}`")

    return df_history.agg({"version": "max"}).collect()[0][0]

# COMMAND ----------

def _get_latest_version_special(df_name: str) -> int:
    df_name_split = df_name.split(".")
    df_history = spark.sql(f"DESCRIBE HISTORY `{df_name_split[0]}`.`{df_name_split[1]}`")

    return df_history.agg({"version": "max"}).collect()[0][0]

# COMMAND ----------

def _get_columns_to_compare(df: DataFrame) -> list():
    return [col_name for col_name in df.columns if "final" in col_name]

# COMMAND ----------

def _get_columns_to_compare_special(df: DataFrame, df_2: DataFrame) -> list():
    # words = ["affinity", "score", "perc", "prob"]
    print("ad_interest_affinity_cosmetics" in df.columns)
    print("ad_interest_affinity_cosmetics" in df_2.columns)
    words = ["score", "perc", "prob"]
    column_list = []
    for col_name in (df.columns)[:5]:
        for word in words:
            if word in col_name and col_name in (df_2.columns):
                column_list.append(col_name)
    return column_list[:2]

# COMMAND ----------

def validate_data_between_two_versions(df_name: str, coef: float = 0.1, limit_of_rows: int = 10000) -> tuple():
    """
    Function that should be able to validate if there are any discrepancies between last two versions of calculated Feature Store rows - bigger than specific thrash-hold

    Parameters:
    df_name: Name of table that is specified in config file
    coef: Thrash-hold under which we expect that rows are calculated just fine
    """

    # load table using YAML
    if "." not in df_name:
        df_path = get_value_from_yaml("paths", df_name)
        latest_version = _get_latest_version(df_path)

        df_1 = spark.read.format("delta").option("versionAsOf", latest_version).load(df_path)
        df_2 = spark.read.format("delta").option("versionAsOf", latest_version - 1).load(df_path)

        columns_to_validate = _get_columns_to_compare(df_1)

    # loading table from table catalog
    else:
        latest_version = _get_latest_version_special(df_name)

        df_1 = spark.read.option("versionAsOf", latest_version).table(df_name)
        df_2 = spark.read.option("versionAsOf", latest_version - 1).table(df_name)
        print(df_1.columns)
        print(df_2.columns)

        columns_to_validate = _get_columns_to_compare_special(df_1, df_2)

    # get random sample of last version table
    df_1_s = df_1.sample(False, 0.1).limit(limit_of_rows)

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

    return (True if df_diff.count() == 0 else False, df_diff.drop("diff"))

# COMMAND ----------

for tab in tables_to_validate:
    print("------------------------")
    print(tab)
    x, y = validate_data_between_two_versions(tab, 0.05, 20000)
    print(x)
    display(y)

# COMMAND ----------

for tab in special_tables_to_validate:
    print("------------------------")
    print(tab)
    x, y = validate_data_between_two_versions(tab, 0.05, 1000)
    print(x)
    display(y)
