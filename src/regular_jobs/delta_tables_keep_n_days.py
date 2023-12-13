# Databricks notebook source
# MAGIC %run ./../../app/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports & Settings

# COMMAND ----------

# TODO this should not be needed
import daipe as dp
from datalakebundle.table.parameters.TableParametersManager import TableParametersManager
from pyspark.sql import SparkSession
from logging import Logger

# COMMAND ----------

# MAGIC %md
# MAGIC ### Keep n days of history

# COMMAND ----------

@dp.notebook_function(
    "%jobs_config.regular_optimization.tables_options%",
    "%jobs_config.regular_optimization.keep_history_n_days%",
)
def bronze_keep_n_days(
    tables_options,
    n_days,
    logger: Logger,
    table_parameters_manager: TableParametersManager,
    spark: SparkSession
):
    for table_name in tables_options:
        if tables_options[table_name]['keep_history']:
            try:
                # extract full table name
                full_table_name = table_parameters_manager.get_or_parse(table_name).full_table_name
                # date column
                date_column = tables_options[table_name]['date_column']

                # minimal date to be kept in table
                min_date = spark.sql(f"SELECT date_add(max({date_column}), -{n_days}) from {full_table_name}").collect()[0][0]

                # perform delete
                spark.sql(f"DELETE FROM {full_table_name} WHERE {date_column} < '{min_date}'")

                logger.info(f"Deleting old records from table {table_name}. New minimal date in table is: {min_date}")
            except BaseException as e:
                logger.error(f"ERROR: Can`t delete records from {table_name} to keep only {n_days} days of history, {e}")
