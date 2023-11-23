# Databricks notebook source
# MAGIC %md Imports

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)

# COMMAND ----------

# MAGIC %md Parameters from widgets

# COMMAND ----------

dbutils.widgets.dropdown("APP_ENV", "dev", ["dev", "prod"])
dbutils.widgets.text("number_of_rows", "")
dbutils.widgets.text("sample_coef", "")

# COMMAND ----------

APP_ENV = dbutils.widgets.get("APP_ENV")

NUMBER_OF_ROWS = dbutils.widgets.get("number_of_rows")
if NUMBER_OF_ROWS in ["None", ""]:
    NUMBER_OF_ROWS = None
else:
    NUMBER_OF_ROWS = int(NUMBER_OF_ROWS)

SAMPLE_COEF = dbutils.widgets.get("sample_coef")
if SAMPLE_COEF in ["None", ""]:
    SAMPLE_COEF = None
else:
    SAMPLE_COEF = float(SAMPLE_COEF)

# COMMAND ----------

# MAGIC %md Define tables and their paths

# COMMAND ----------

# Specify path into "old" data from non-up-to-date-odap pipeline
tables_to_validate = {
    "income_interest_scores": f"abfss://silver@cpexstorageblob{APP_ENV}.dfs.core.windows.net/income/income_interest_scores.delta",
    # "income_interest_coeffs",
    "income_url_scores": f"abfss://silver@cpexstorageblob{APP_ENV}.dfs.core.windows.net/income/income_url_scores.delta",
    # "income_url_coeffs",
    "income_other_scores": f"abfss://silver@cpexstorageblob{APP_ENV}.dfs.core.windows.net/income/income_other_scores.delta",
    # "income_other_coeffs",
    "education_interest_scores": f"abfss://silver@cpexstorageblob{APP_ENV}.dfs.core.windows.net/education/education_interest_scores.delta",
    # "education_interest_coeffs",
    "education_url_scores": f"abfss://silver@cpexstorageblob{APP_ENV}.dfs.core.windows.net/education/education_url_scores.delta",
    # "education_url_coeffs",
    "education_other_scores": f"abfss://silver@cpexstorageblob{APP_ENV}.dfs.core.windows.net/education/education_other_scores.delta",
    # "education_other_coeffs",
    # "user_entity_fs":f"abfss://gold@cpexstorageblob{APP_ENV}.dfs.core.windows.net/feature_store/features/user_entity",
}

# COMMAND ----------

special_tables_to_validate = {
    "odap_features_user.latest": f"abfss://gold@cpexstorageblob{APP_ENV}.dfs.core.windows.net/feature_store/features/user_entity.delta",
    # "odap_features_user.user",
    # "odap_features_user.user_stage1",
    # "odap_features_user.user_stage2",
    # "odap_features_user.user_stage3",
}

# COMMAND ----------

# MAGIC %md Helper functions

# COMMAND ----------


def _get_columns_to_compare(df: DataFrame) -> []:
    return [col_name for col_name in df.columns if "final" in col_name]


# COMMAND ----------


def _get_columns_to_compare_special(df: DataFrame, df_2: DataFrame) -> []:
    # words = ["affinity", "score", "perc", "prob"]
    # print("ad_interest_affinity_cosmetics" in df.columns)
    # print("ad_interest_affinity_cosmetics" in df_2.columns)
    words = ["score", "perc", "prob"]
    column_list = []
    for col_name in (df.columns)[:5]:
        for word in words:
            if word in col_name and col_name in (df_2.columns):
                column_list.append(col_name)
    return column_list[:2]


# COMMAND ----------

# MAGIC %md Main function

# COMMAND ----------


def validate_data_between_two_sources(
    df_name: str,
    df_path_old_pipeline: str,
    coef: float = 0.1,
    limit_of_rows: int = 10000,
) -> tuple():
    """
    Function that should be able to validate if there are any discrepancies between last
     two versions of calculated Feature Store rows - bigger than specific thrash-hold

    Parameters:
    df_name: Name of table that is specified in config file
    coef: Thrash-hold under which we expect that rows are calculated just fine
    """

    coef = 0.1 if coef is None else coef  # handle 'None' input for coef

    # load table using YAML
    if "." not in df_name:
        df_path = get_value_from_yaml("paths", df_name)

        df_1 = spark.read.format("delta").load(
            df_path
        )  # read "current" ODAP pipeline results
        df_2 = spark.read.format("delta").load(
            df_path_old_pipeline
        )  # read "old" pipeline results

        columns_to_validate = _get_columns_to_compare(df_1)

    # loading table from table catalog
    else:
        df_1 = spark.read.table(df_name)  # read "current" ODAP pipeline results
        df_2 = spark.read.format("delta").load(
            df_path_old_pipeline
        )  # read "old" pipeline results

        columns_to_validate = _get_columns_to_compare_special(df_1, df_2)

    # get random sample of last version table
    if limit_of_rows is None:
        df_1_s = df_1.sample(False, 0.1)
    else:
        df_1_s = df_1.sample(False, 0.1).limit(limit_of_rows)

    df_join = (
        df_1_s.alias("df_1")
        .join(df_2.alias("df_2"), on="user_id", how="inner")
        .withColumn("diff", F.lit(0))
    )  # add column to check validity and for further filtering

    df_diff = df_join.withColumn("diff", F.lit(0))

    for col in columns_to_validate:
        df_diff = df_diff.withColumn(
            "diff",
            F.when(
                F.abs(
                    (F.col(f"df_1.{col}") - F.col(f"df_2.{col}")) / F.col(f"df_1.{col}")
                )
                < F.lit(coef),
                F.col("diff"),
            ).otherwise(F.lit(1)),
        )

    print(
        f"There were comparison done for {df_diff.count()} columns, {df_1.count()} from df_1 and {df_2.count()} from df_2"
    )

    df_diff = df_diff.filter(F.col("diff") == F.lit(1))

    print(
        f"There left {df_diff.count()} columns for comparison after removing valid non-different (valid) rows"
    )

    df_diff_zeros = df_diff.select("*")
    for column in columns_to_validate:
        df_diff_zeros = (
            df_diff_zeros
            # it's true that both columns will be zero at same time - good to know how many rows we skipped
            .filter(
                (F.col(f"df_1.{column}") == F.lit(0))
                & (F.col(f"df_2.{column}") == F.lit(0))
            )
        )
        df_diff = (
            df_diff
            # it's not true that both columns will be zero at same time
            .filter(
                ~(F.col(f"df_1.{column}") == F.lit(0))
                & (F.col(f"df_2.{column}") == F.lit(0))
            )
        )

    print(
        f"After cleanse of zeros in columns, there were left different {df_diff.count()} columns"
    )
    print(
        f"After keeping zeros in columns, there were skipped {df_diff_zeros.count()} columns"
    )

    return (df_diff.count() == 0, df_diff.drop("diff"))


# COMMAND ----------

# MAGIC %md Validation itself

# COMMAND ----------

# No need to run, these two "sources" link the same data
# for tab, path in tables_to_validate.items():
#     print("------------------------")
#     print(tab)
#     x, y = validate_data_between_two_sources(tab, path, SAMPLE_COEF, NUMBER_OF_ROWS)
#     print(x)
#     display(y)  # pylint: disable=undefined-variable

# COMMAND ----------

for tab, path in special_tables_to_validate.items():
    print("------------------------")
    print(tab)
    x, y = validate_data_between_two_sources(tab, path, SAMPLE_COEF, NUMBER_OF_ROWS)
    print(x)
    display(y)  # pylint: disable=undefined-variable
