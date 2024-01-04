# Databricks notebook source
# MAGIC %run ./../../app/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports & Settings

# COMMAND ----------

from src.utils.read_config import config
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger

# COMMAND ----------

logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimize

# COMMAND ----------

tables_options = config.jobs_config.regular_optimization.tables_options
for table_name in tables_options:
    if tables_options[table_name]['optimize']:
        try:
            # firstly get full table name (e.g. dev_bronze.table_name)
            full_table_path = config.paths.get(table_name)
                
            spark.sql(f"OPTIMIZE {full_table_path}")

            logger.info(f"Optimizing table: {table_name}")
        except BaseException as e:
            logger.error(f"ERROR: Can`t OPTIMIZE {table_name}, {e}")

              
