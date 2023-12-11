# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #Rename interests
# MAGIC This notebook serves to rename an existing interest name in the Adpicker app. Intended for interactive use.

# COMMAND ----------

# MAGIC %run ./../../../app/bootstrap

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Imports

# COMMAND ----------

# TODO this should not be needed
import daipe as dp

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Read data

# COMMAND ----------

@dp.transformation(dp.read_delta('%interests.delta_path%'))
def read_interests(df):
    return df 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Find ID of interest to rename

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC You can optionally use this function to find the ID of subinterest you want to rename.
# MAGIC 
# MAGIC (use `interest_pattern` variable & **SQL-like pattern** to find the interest).

# COMMAND ----------

interest_pattern = ""

# COMMAND ----------

@dp.transformation(read_interests)
def find_id(df):
    df.filter(F.col("subinterest").like(interest_pattern)).display()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Inputs
# MAGIC 
# MAGIC Fill ID of interest to rename below:

# COMMAND ----------

db_id = None # FILL INTEREST ID

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Before change
# MAGIC 
# MAGIC Displaying the interest row before any changes. 
# MAGIC 
# MAGIC #### Make sure the following query returns one single correct row!

# COMMAND ----------

@dp.transformation(read_interests)
def check_current_name(df):
    assert db_id is not None, "ID has to be defined."
    assert db_id in df.rdd.map(lambda x: x.db_id).collect(), "Nonexistent interest ID."
    df_filtered = df.filter(F.col("db_id") == db_id)
    assert df_filtered.count() == 1, f"{df_filtered.count()} rows returned instead of 1."
    df_filtered.display()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Renaming
# MAGIC 
# MAGIC Choose new subinterest name and rename the in-application name (db_subinterest_name) in delta. Check if the outcome of renaming is correct.

# COMMAND ----------

new_db_subinterest_name = None

# COMMAND ----------

@dp.transformation(read_interests, display=False)
def interests_renamed(df: DataFrame):
    assert db_id is not None, "ID has to be defined."
    assert db_id in df.rdd.map(lambda x: x.db_id).collect(), "Nonexistent interest ID."
    assert new_db_subinterest_name, "New subinterest name cannot be none - fill the name."
    df = (
        df
        .withColumn('db_subinterest', F.when(F.col("db_id") == db_id, new_db_subinterest_name).otherwise(F.col("db_subinterest")))
    )
    df.filter(F.col("db_id") == db_id).display()
    return df

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Update
# MAGIC 
# MAGIC If the renaming was succesful, update the interest delta table:

# COMMAND ----------

@dp.transformation(interests_renamed)
@dp.delta_overwrite('%interests.delta_path%')
def save_interest_table(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Done!
# MAGIC 
# MAGIC Don't forget to run the *update_db* notebook to write the changes into the MySQL database.
