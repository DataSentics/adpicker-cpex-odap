import datetime as dt
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark
import builtins as py
import shutil
from pyspark.sql import SQLContext
from pyspark.dbutils import DBUtils

# pylint: disable=protected-access

def upsertToDelta(aggregateTable, updateTable, key, sc):
    # Set the dataframe to view name
    aggregateTable.createOrReplaceTempView("aggregates")
    updateTable.createOrReplaceTempView("updates")
    sqlContext = SQLContext(sc)
    # Use the view name to apply MERGE
    # NOTE: You have to use the SparkSession that has been used to define the `updates` dataframe
    df = sqlContext.sql(
        "MERGE INTO aggregates t USING updates s ON s."
        + key
        + " = t."
        + key
        + " WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"
    )
    return df


def check_dbfs_existence(path: str, spark):
    import re

    filePath = re.sub("^/dbfs/", "dbfs:/", path)
    jvm = spark._jvm
    jsc = spark._jsc
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(jsc.hadoopConfiguration())
    return fs.exists(jvm.org.apache.hadoop.fs.Path(filePath))


def create_latest_lookalike_table(db, lookalikes, spark):
    df_tmp_int = spark.createDataFrame(
        [(dt.date.today().isoformat(), lookalikes)], ["date", "last_used_lookalikes"]
    )
    df_tmp_int.createOrReplaceTempView("last_used_lookalikes")
    spark.sql(
        f"CREATE TABLE {db}.last_used_lookalikes as SELECT * FROM last_used_lookalikes"
    )
    spark.catalog.dropTempView("last_used_lookalikes")


# Get old interests and identify new ones, update table if necessary
def create_latest_interests_table(db, interests, spark):
    df_tmp_int = spark.createDataFrame(
        [(dt.date.today().isoformat(), interests)], ["date", "last_used_interests"]
    )
    df_tmp_int.createOrReplaceTempView("last_used_interests")
    spark.sql(
        f"CREATE TABLE {db}.last_used_interests as SELECT * FROM last_used_interests"
    )
    spark.catalog.dropTempView("last_used_interests")


def delete_checkpoint_dir(sc):
    """
    Delete checkpoint directory with its' content.
    Use at the end of a notebook using checkpoints.
    """
    checkpointDir = sc._jsc.sc().getCheckpointDir().get().replace("dbfs:", "/dbfs")
    print("DELETING CHECKPOINT DIRECTORY: ", checkpointDir)
    shutil.rmtree(checkpointDir, ignore_errors=True)
