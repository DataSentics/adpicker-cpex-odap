# Databricks notebook source
# MAGIC %md ###Deleting

# COMMAND ----------

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
    N_DAYS_TO_KEEP = max(N_DAYS_TO_KEEP, 60)
except (TypeError, ValueError) as e:
    logger.error(
        "Wrongly entered value, check if is it 'int' and if is it NOT empty. Check %s",
        e,
    )
    raise

# COMMAND ----------

fs_path = f"abfss://gold@cpexstorageblob{environment}.dfs.core.windows.net/feature_store/features/user_entity.delta"

# COMMAND ----------


def featurestore_keep_n_days(full_table_path: str, n_days: int):
    try:
        # date column
        date_column = "timestamp"

        # minimal date to be kept in table
        min_date = spark.sql(
            f"SELECT date_add(max({date_column}), -{n_days}) from delta.`{full_table_path}`"
        ).collect()[0][0]

        # perform delete
        spark.sql(
            f"DELETE FROM delta.`{full_table_path}` WHERE {date_column} < '{min_date}'"
        )

        logger.info(
            "Deleting old records from table %s. New minimal date in table is: %s",
            full_table_path,
            min_date,
        )
    except AnalysisException as e:  # pylint: disable=redefined-outer-name
        logger.error(
            "ERROR: Can`t delete records from %s to keep only %s days of history, %s",
            full_table_path,
            n_days,
            e,
        )
        raise


# COMMAND ----------

# do deletion itself
featurestore_keep_n_days(fs_path, N_DAYS_TO_KEEP)
