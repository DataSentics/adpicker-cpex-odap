# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Update DB
# MAGIC 
# MAGIC This notebook reads the delta file with interest definitions and uses it to overwrite (update) the definitions in a MySQL DB used by the app.

# COMMAND ----------

# MAGIC %run ./../../../app/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Imports

# COMMAND ----------

# global imports
import daipe as dp

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.session import SparkSession
from pyspark.dbutils import DBUtils


# project-level imports
from adpickercpex.lib.display_result import display_result

from adpickercpex.solutions._DB_connection_functions import overwrite_mysql_table_by_df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Load interests

# COMMAND ----------

@dp.transformation(dp.read_delta('%interests.delta_path%'), display=False)
@display_result()
def read_interests(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Processing
# MAGIC 
# MAGIC Minor changes to the interests DF (necessary to keep the DB consistent).

# COMMAND ----------

@dp.transformation(read_interests, display=False)
def process_interests(df: DataFrame):
    # dtypes
    df = df.withColumn('db_id', F.col('db_id').cast('int'))
    # encoding as JSON strings
    for c in ("keywords", "keywords_bigrams"):
        df = df.withColumn(c, F.to_json(c))
    # replacing empty values with nulls
    for c in ("domains", "keywords", "keywords_bigrams"):
        df = df.withColumn(c, F.when(F.col(c) == "[]", None).otherwise(F.col(c)))
           
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## DB update
# MAGIC 
# MAGIC Overwritting the database:

# COMMAND ----------

@dp.notebook_function(process_interests)
def update_db(df: DataFrame, spark: SparkSession, dbutils: DBUtils):
    df = (df
          .select(
              F.col('db_id').alias('id'),
              F.col('db_subinterest').alias('name'),
              F.lit(None).alias('client_id').cast('integer'),
              'keywords',
              'keywords_bigrams',
              'domains',
              'active',
              'visible',
              F.col('db_general_interest').alias('general_interest')
          )
         )

    overwrite_mysql_table_by_df(df, 'interests', spark, dbutils)
