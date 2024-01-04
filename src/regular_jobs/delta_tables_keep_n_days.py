# Databricks notebook source
# MAGIC %run ./../../app/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports & Settings

# COMMAND ----------

from src.utils.helper_functions_defined_by_user.logger import instantiate_logger
from src.utils.read_config import config

# COMMAND ----------

# MAGIC %md
# MAGIC ### Keep n days of history

# COMMAND ----------
logger = instantiate_logger()
# COMMAND ----------

n_days = config.jobs_config.regular_optimization.keep_history_n_days
tables_options = config.jobs_config.regular_optimization.tables_options
for table_name in tables_options:
    if tables_options[table_name]['keep_history']:
        try:
            # extract full table name
            full_table_path = config.paths.get(table_name)
            # date column
            date_column = tables_options[table_name]['date_column']
            # minimal date to be kept in table
            min_date = spark.sql(f"SELECT date_add(max({date_column}), -{n_days}) from {full_table_path}").collect()[0][0]
            # perform delete
            spark.sql(f"DELETE FROM {full_table_path} WHERE {date_column} < '{min_date}'")

            logger.info(f"Deleting old records from table {table_name}. New minimal date in table is: {min_date}")
        except BaseException as e:
            logger.error(f"ERROR: Can`t delete records from {table_name} to keep only {n_days} days of history, {e}")

