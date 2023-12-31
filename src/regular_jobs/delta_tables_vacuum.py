# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports & Settings

# COMMAND ----------

from src.utils.read_config import config
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger

# COMMAND ----------

logger = instantiate_logger()

# COMMAND ----------

vacuum_time = config.jobs_config.regular_optimization.vacuum_keep_n_hours
tables_options = config.jobs_config.regular_optimization.tables_options
for table_name in tables_options:
    if tables_options[table_name]['vacuum']:
        try:
            # get full table path
            full_table_path = config.paths.get(table_name)
            # perform vacuum
            spark.sql(f"VACUUM delta.`{full_table_path}` RETAIN {vacuum_time} HOURS")
            # perform vacuum
            logger.info("Vacuuming table: %s", table_name)
        except BaseException as e:
            logger.error("ERROR: Can`t vacuum %s, %s", table_name, e)
