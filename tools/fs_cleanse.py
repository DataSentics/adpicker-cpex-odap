# Databricks notebook source
# MAGIC %md ###Deleting

# COMMAND ----------

# MAGIC %md Parameters from widgets

# COMMAND ----------

dbutils.widgets.dropdown("APP_ENV", "dev", ["dev", "prod"])
dbutils.widgets.text("n_days_to_keep", "")
dbutils.widgets.text("n_days_to_vacuum", "")

# COMMAND ----------

APP_ENV = dbutils.widgets.get("APP_ENV")

N_DAYS_TO_KEEP = dbutils.widgets.get("n_days_to_keep")
if N_DAYS_TO_KEEP in ["None", ""]:
    N_DAYS_TO_KEEP = None
else:
    N_DAYS_TO_KEEP = int(N_DAYS_TO_KEEP)

N_DAYS_TO_VACUUM = dbutils.widgets.get("n_days_to_vacuum")
if N_DAYS_TO_VACUUM in ["None", ""]:
    N_DAYS_TO_VACUUM = None
else:
    N_DAYS_TO_VACUUM = int(N_DAYS_TO_VACUUM)

# COMMAND ----------

fs_path = f"abfss://gold@cpexstorageblob{APP_ENV}.dfs.core.windows.net/feature_store/features/user_entity.delta"

# COMMAND ----------


def featurestore_keep_n_days(full_table_path: str, n_days: int):
    try:
        # keep at least 60 days
        if n_days < 60 or n_days is None:
            n_days = 60

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

        print(
            f"Deleting old records from table {full_table_path}. New minimal date in table is: {min_date}"
        )
    except BaseException as e:
        print(
            f"ERROR: Can`t delete records from {full_table_path} to keep only {n_days} days of history, {e}"
        )


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
featurestore_perform_vacuum(fs_path, N_DAYS_TO_VACUUM)

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY delta.`abfss://gold@cpexstorageblobdev.dfs.core.windows.net/feature_store/features/user_entity.delta`
