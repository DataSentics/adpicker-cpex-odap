# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Update existing interest
# MAGIC 
# MAGIC This notebook allows to update the delta table definition of an already existing interest.

# COMMAND ----------

# MAGIC %run ./../../../app/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Imports

# COMMAND ----------

# global imports
# TODO this should not be needed
import daipe as dp
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F

from logging import Logger


# project-level imports
from src.utils.helper_functions_defined_by_user._functions_helper import (
    add_binary_flag_widget,
    check_binary_flag_widget, 
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Settings

# COMMAND ----------

@dp.notebook_function()
def create_widgets(widgets: dp.Widgets):
    widgets.add_text("interest_name", "", "Interest name")
    
    add_binary_flag_widget(widgets, name="update_keywords")
    add_binary_flag_widget(widgets, name="update_keywords_bigrams", label="Update keywords bigrams")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Insert the new version here:
# MAGIC 
# MAGIC single example: `["token_1", "token_2", "token_3"]`
# MAGIC bigrams example: `["token one", "token two", "token three"]`

# COMMAND ----------

@dp.notebook_function()
def keywords_single():
    return [
    ]

# COMMAND ----------

@dp.notebook_function()
def keywords_bigrams():
    return [
    ]

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Read data
# MAGIC 
# MAGIC Load interests:

# COMMAND ----------

@dp.transformation(dp.read_delta('%interests.delta_path%'), display=False)
def read_interests(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Update
# MAGIC 
# MAGIC Update of the interests DF:

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Single words:

# COMMAND ----------

@dp.transformation(read_interests, keywords_single, dp.get_widget_value('interest_name'), dp.get_widget_value('update_keywords'), display=False)
def update_keywords_single(df: DataFrame, keywords, interest_name, update_keywords, logger: Logger):
    if check_binary_flag_widget(update_keywords):
        df = (df
              .withColumn('keywords', 
                          F.when(F.col('subinterest') == interest_name, F.array(*[F.lit(kw) for kw in keywords]))
                          .otherwise(F.col('keywords')))
             )
        logger.info(f"Updating single-word keywords for `{interest_name}`. New definition size: {len(keywords)}")

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Bigrams:

# COMMAND ----------

@dp.transformation(update_keywords_single, keywords_bigrams, dp.get_widget_value('interest_name'), dp.get_widget_value('update_keywords_bigrams'), display=False)
def update_keywords_bigrams(df: DataFrame, keywords, interest_name, update_keywords, logger: Logger):
    if check_binary_flag_widget(update_keywords):
        df = (df
              .withColumn('keywords_bigrams', 
                          F.when(F.col('subinterest') == interest_name, F.array(*[F.lit(kw) for kw in keywords]))
                          .otherwise(F.col('keywords_bigrams')))
             )
        logger.info(f"Updating bigram keywords for `{interest_name}`. New definition size: {len(keywords)}")

    return df

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Save
# MAGIC 
# MAGIC Overwrite table with the new version:

# COMMAND ----------

@dp.transformation(update_keywords_bigrams, display=False)
@dp.delta_overwrite('%interests.delta_path%')
def save_interests(df: DataFrame):
    return df
