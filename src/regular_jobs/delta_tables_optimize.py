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
# MAGIC ### Optimize

# COMMAND ----------

@dp.notebook_function("%jobs_config.regular_optimization.tables_options%")
def perform_optimize(
    tables_options,
    logger: Logger,
    table_parameters_manager: TableParametersManager,
    spark: SparkSession
):
    for table_name in tables_options:
        if tables_options[table_name]['optimize']:
            try:
                # firstly get full table name (e.g. dev_bronze.table_name)
                full_table_name = table_parameters_manager.get_or_parse(table_name).full_table_name
                spark.sql(f"OPTIMIZE {full_table_name}")

                logger.info(f"Optimizing table: {table_name}")
            except BaseException as e:
                logger.error(f"ERROR: Can`t OPTIMIZE {table_name}, {e}")
