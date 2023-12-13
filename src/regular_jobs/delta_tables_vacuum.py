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
# MAGIC ### Perform vacuum

# COMMAND ----------

@dp.notebook_function(
    "%jobs_config.regular_optimization.tables_options%",
    "%jobs_config.regular_optimization.vacuum_n_hours%",
)
def perform_vacuum(
    tables_options,
    vacuum_time,
    logger: Logger,
    table_parameters_manager: TableParametersManager,
    spark: SparkSession,
):
    for table_name in tables_options:
        if tables_options[table_name]['vacuum']:
            try:
                # firstly get full table name (e.g. dev_bronze.table_name)
                full_table_name = table_parameters_manager.get_or_parse(table_name).full_table_name
                # perform vacuum
                spark.sql(f"VACUUM {full_table_name} RETAIN {vacuum_time} HOURS")

                logger.info(f"Vacuuming table: {table_name}")
            except BaseException as e:
                logger.error(f"ERROR: Can`t vacuum {table_name}, {e}")
