# Databricks notebook source
# MAGIC %md ###Deleting

# COMMAND ----------

import os
import logging
from pyspark.sql.utils import AnalysisException

# COMMAND ----------

logger = logging.getLogger("py4j")
logger.setLevel(logging.ERROR)
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md Parameters from widgets

# COMMAND ----------

dbutils.widgets.text("n_hours_to_vacuum", "")

# COMMAND ----------

environment = os.environ["APP_ENV"]

try:
    N_HOURS_TO_VACUUM = int(dbutils.widgets.get("n_hours_to_vacuum"))
    # keep at least 20 days (480 hours)
    N_HOURS_TO_VACUUM = max(N_HOURS_TO_VACUUM, 480)
except (TypeError, ValueError) as e:
    logger.error(
        "Wrongly entered value, check if is it 'int' and if is it NOT empty. Check %s",
        e,
    )
    raise

# COMMAND ----------

fs_path = f"abfss://gold@cpexstorageblob{environment}.dfs.core.windows.net/feature_store/features/user_entity.delta"

# COMMAND ----------


def featurestore_perform_vacuum(full_table_path: str, vacuum_time: int):
    try:
        # perform vacuum
        spark.sql(f"VACUUM delta.`{full_table_path}` RETAIN {vacuum_time} HOURS")

        logger.info("Vacuuming table: %s", full_table_path)
    except AnalysisException as e:  # pylint: disable=redefined-outer-name
        logger.error(
            "ERROR: Can`t vacuum %s, check following error: %s", full_table_path, e
        )
        raise


# COMMAND ----------

# do vacuuming itself
featurestore_perform_vacuum(fs_path, N_HOURS_TO_VACUUM)
