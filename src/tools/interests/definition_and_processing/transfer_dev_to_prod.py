# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Transfer dev to prod
# MAGIC 
# MAGIC This notebook updates the PROD version of interests with the version from the DEV.
# MAGIC 
# MAGIC It should be used to add new interest, update existing one, etc.:
# MAGIC * Firts, update the version on DEV
# MAGIC * Second, transfer the version to PROD

# COMMAND ----------

# MAGIC %run ./../../../app/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Imports

# COMMAND ----------

import daipe as dp
from pyspark.sql.dataframe import DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Definitions

# COMMAND ----------

@dp.notebook_function('%interests.delta_path%', '%kernel.environment%')
def get_paths(interests_path, environment):
    if environment == 'dev':
        return {
            'dev_path': interests_path,
            'prod_path': interests_path.replace('dev', 'prod'),
        }
    return {
        'dev_path': interests_path.replace('prod', 'dev'),
        'prod_path': interests_path,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Transfer

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Load the table from DEV:

# COMMAND ----------

@dp.transformation(dp.read_delta(get_paths.result['dev_path']), display=False)
def load_from_dev(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Save it to PROD:

# COMMAND ----------

@dp.transformation(load_from_dev, display=False)
@dp.delta_overwrite(get_paths.result['prod_path'])
def save_to_prod(df: DataFrame):
    return df
