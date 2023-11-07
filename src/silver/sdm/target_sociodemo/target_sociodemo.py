# Databricks notebook source
# MAGIC %md
# MAGIC #### Imports

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame
from datetime import date, datetime, timedelta

from src.utils.helper_functions_defined_by_user.table_writing_functions import (
    write_dataframe_to_table,
)
from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger


from src.schemas.sociodemo_target_schema import get_sociodemo_target_schema

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Widgets

# COMMAND ----------

dbutils.widgets.text("end_date", "", "End date")
dbutils.widgets.text("n_days", "7", "Number of days to include")

# COMMAND ----------

widget_end_date = dbutils.widgets.get("end_date")
widget_n_days = dbutils.widgets.get("n_days")

# COMMAND ----------


# if no end date provided then current date is taken
def load_silver(df: DataFrame, end_date: str, n_days: str):
    # process end date
    try:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    except BaseException:
        end_date = date.today()

    # calculate start date
    start_date = end_date - timedelta(days=int(n_days))
    # to deal for potential minor mismatch between day (from file upload) and DATE (extracted from EVENT_TIME) column
    start_date_safe = end_date - timedelta(days=int(n_days) + 2)

    return (
        df.filter(F.col("day") >= start_date_safe)
        .select(
            "AGE",
            "GENDER",
            F.col("DEVICE").alias("USER_ID"),
            F.col("OWNER_NAME").alias("PUBLISHER"),
            F.col("EVENT_TIME").alias("TIMESTAMP"),
            F.to_date(F.col("EVENT_TIME")).alias("DATE"),
        )
        .filter((F.col("DATE") >= start_date) & (F.col("DATE") <= end_date))
    )


df_bronze_cpex_piano = spark.read.format("delta").load(
    get_value_from_yaml("paths", "cpex_table_piano")
)
df_silver_cpex_piano = load_silver(df_bronze_cpex_piano, widget_end_date, widget_n_days)

# COMMAND ----------

# MAGIC %md Process age and gender

# COMMAND ----------


def preprocessing(df: DataFrame):
    return (
        df.withColumn("AGE", F.col("AGE").cast("int"))
        .withColumn(
            "AGE",
            F.when((F.col("AGE") < 0) | (F.col("AGE") > 100), None).otherwise(
                F.col("AGE")
            ),
        )
        .withColumn("AGE", F.col("AGE").cast("string"))
        .withColumn(
            "AGE", F.when(F.col("AGE").isNull(), "unknown").otherwise(F.col("AGE"))
        )
        # 1 = female, 2 = male but in piano 1 = male
        .withColumn(
            "GENDER",
            F.when((F.col("GENDER").isNull()) | (F.col("GENDER") == "0"), "unknown")
            .when(F.col("GENDER") == "1", "2")
            .when(F.col("GENDER") == "2", "1")
            .otherwise(F.col("GENDER")),
        )
    )


df_preprocessing = preprocessing(df_silver_cpex_piano)

# COMMAND ----------

# MAGIC %md Save table

# COMMAND ----------

schema, info = get_sociodemo_target_schema()

write_dataframe_to_table(
    df_preprocessing,
    get_value_from_yaml("paths", "sdm_sociodemo_targets"),
    schema,
    "overwrite",
    root_logger,
    info["partition_by"],
    info["table_properties"],
)
