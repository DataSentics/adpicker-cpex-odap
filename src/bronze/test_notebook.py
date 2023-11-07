# Databricks notebook source
from src.utils.helper_functions_defined_by_user.table_writing_functions import (
    write_dataframe_to_table,
)

import logging
import pyspark.sql.types as T

# COMMAND ----------


def instantiate_logger():
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a formatter with timestamp
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create a stream handler and set its formatter
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Add the stream handler to the logger
    logger.addHandler(stream_handler)
    return logger


logger = instantiate_logger()

# COMMAND ----------

dest = "/mnt/aam-cpex-dev/solutions/testing/odler_data_test.delta"
one = T.StructType(
    [
        T.StructField("value", T.LongType(), True),
        T.StructField("name", T.StringType(), False),
    ]
)
two = T.StructType(
    [
        T.StructField("name", T.StringType(), True),
        T.StructField("value", T.LongType(), False),
    ]
)
df = spark.createDataFrame([["Michal", 6], ["Nicole", 12], ["Alfonzo", 100]], two)


# COMMAND ----------

dest = "/mnt/aam-cpex-dev/silver/user_traits.delta"
df = spark.read.format("delta").load(dest)
df.printSchema()

# COMMAND ----------

df_table = spark.read.format("delta").load(dest)
# df.display()
# df.printSchema()

# df.write.format("delta").mode("append").save(dest)
df_table.printSchema()

# COMMAND ----------

# write_dataframe_to_table(df, dest, schema, "overwrite", logger)


# COMMAND ----------

import os
import yaml

path = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
print(os.path.abspath(path))
new_path = "/Workspace/Repos/michal.odler@datasentics.com/adpicker-cpex-odap/src/config/config.yaml"

with open(new_path, "r", encoding="utf-8") as cf:
    data = yaml.safe_load(cf)
    print(data)

# COMMAND ----------

df = spark.read.format("delta").load(
    "/mnt/aam-cpex-dev/solutions/testing/cpex_piano.delta"
)
df_app = df.drop("browser").limit(10)
dest = "/mnt/aam-cpex-dev/solutions/testing/cpex_piano.delta"

# COMMAND ----------
