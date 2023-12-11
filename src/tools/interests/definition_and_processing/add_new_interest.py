# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Add new interest
# MAGIC 
# MAGIC This notebook allows to add new interest definition to the main delta table.

# COMMAND ----------

# MAGIC %run ./../../../app/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Imports

# COMMAND ----------

import daipe as dp

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import types as T
from pyspark.sql.session import SparkSession
from pyspark.sql import functions as F


# TODO this should not be needed
from adpickercpex.lib.display_result import display_result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load current interest table

# COMMAND ----------

@dp.transformation(dp.read_delta('%interests.delta_path%'), display=False)
def read_interests(df):
    return df

# COMMAND ----------

last_db_id = read_interests_df.select(F.max("db_id")).collect()[0][0]
print(f"Last interest id is {last_db_id}")

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Inputs
# MAGIC 
# MAGIC Manually define new interest below, `db_id` is calculated automatically:

# COMMAND ----------

subinterest_name = "interest_name_here"
general_interest_name = "general_interest_name_here"

db_subinterest_name = "Interest name here"
db_general_interest_name = "General interest name here"

db_id = last_db_id + 1
print(f"New interest id wil be {db_id}")

# COMMAND ----------

keywords = []

# COMMAND ----------

keywords_bigrams = []

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Before change
# MAGIC 
# MAGIC Displaying the interest table before any changes.

# COMMAND ----------

read_interests_df.orderBy("db_id", ascending=False).display()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Preparation
# MAGIC 
# MAGIC Preparation of the new row with the interest:

# COMMAND ----------

@dp.transformation(display=False)
@display_result()
def new_interest(spark: SparkSession):
    assert (" " not in subinterest_name and  " " not in general_interest_name), \
        "subinterest_name and general_interest_name must not contain spaces"
    
    data = [
        [
            db_id,
            subinterest_name,
            general_interest_name,
            keywords,
            keywords_bigrams,
            db_general_interest_name,
            db_subinterest_name,
            f"ad_interest_affinity_{subinterest_name.lower()}",
            None,
            True,
            True,
        ]
    ]

    schema = T.StructType([
       T.StructField("db_id", T.LongType(), True),
       T.StructField("subinterest", T.StringType(), True),
       T.StructField("general_interest", T.StringType(), True),
       T.StructField("keywords", T.ArrayType(T.StringType(), True), True),
       T.StructField("keywords_bigrams", T.ArrayType(T.StringType(),True), True),
       T.StructField("db_general_interest", T.StringType(), True),
       T.StructField("db_subinterest", T.StringType(), True),
       T.StructField("subinterest_fs", T.StringType(), True),
       T.StructField("domains", T.StringType(), True),
       T.StructField("active", T.BooleanType(), True),
       T.StructField("visible", T.BooleanType(), True),
    ])

    return spark.createDataFrame(data, schema=schema)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Update
# MAGIC 
# MAGIC Update of the delta table:

# COMMAND ----------

@dp.transformation(new_interest, display=False)
@dp.delta_append('%interests.delta_path%')
def update_interests(df: DataFrame):
    return df

# COMMAND ----------


