# Databricks notebook source
# MAGIC %md ###Deleting

# COMMAND ----------

import pyspark.sql.functions as F
import os
import logging
from pyspark.sql.utils import AnalysisException

# COMMAND ----------

# MAGIC %md Parameters from widgets

# COMMAND ----------

dbutils.widgets.text("n_days_to_keep", "")

# COMMAND ----------

# MAGIC %md Setup logger

# COMMAND ----------

logger = logging.getLogger("py4j")
logger.setLevel(logging.ERROR)
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

environment = os.environ["APP_ENV"]

try:
    N_DAYS_TO_KEEP = int(dbutils.widgets.get("n_days_to_keep"))
    # keep at least 60 days
    if N_DAYS_TO_KEEP < 60:
        N_DAYS_TO_KEEP = 60
except TypeError, ValueError as e:
    logger.error(f"Wrongly entered value, check if is it 'int' and if is it NOT empty. Check {e}")

# COMMAND ----------

fs_path = f"abfss://gold@cpexstorageblob{environment}.dfs.core.windows.net/feature_store/features/user_entity.delta"

# COMMAND ----------

def featurestore_keep_n_days(full_table_path: str, n_days: int):
    try:
        # date column
        date_column = "timestamp"

        # minimal date to be kept in table
        min_date = spark.sql(f"SELECT date_add(max({date_column}), -{n_days}) from delta.`{full_table_path}`").collect()[0][0]

        # perform delete
        spark.sql(f"DELETE FROM delta.`{full_table_path}` WHERE {date_column} < '{min_date}'")

        logger.info(f"Deleting old records from table {full_table_path}. New minimal date in table is: {min_date}")
    except AnalysisException as e:
        logger.error(f"ERROR: Can`t delete records from {full_table_path} to keep only {n_days} days of history, {e}")

# COMMAND ----------

# do deletion itself
featurestore_keep_n_days(fs_path, N_DAYS_TO_KEEP)

# COMMAND ----------

# MAGIC %md ###Vacuuming

# COMMAND ----------

def featurestore_perform_vacuum(full_table_path: str, vacuum_time: int):
    try:
        # keep at least 20 days
        if vacuum_time < 480 or vacuum_time is None:
            vacuum_time = 480

        # perform vacuum
        spark.sql(f"VACUUM delta.`{full_table_path}` RETAIN {vacuum_time} HOURS")

        print(f"Vacuuming table: {full_table_path}")
    except BaseException as e:
        print(f"ERROR: Can`t vacuum {full_table_path}, {e}")

# COMMAND ----------

# do vacuuming itself
# featurestore_perform_vacuum(fs_path, N_DAYS_TO_VACUUM)
