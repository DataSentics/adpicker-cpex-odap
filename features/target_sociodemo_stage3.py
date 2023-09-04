# Databricks notebook source
# MAGIC %md #### Imports

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

from src.utils.helper_functions_defined_by_user.yaml_functions import get_value_from_yaml

publishers_list = ['eco']

# COMMAND ----------

dbutils.widgets.dropdown("target_name", "<no target>", ["<no target>"], "01. target name")
dbutils.widgets.text("timestamp", "2023-03-10", "02. timestamp")
dbutils.widgets.dropdown("sample_data", "complete", ["complete", "sample"], "03. sample data")

# COMMAND ----------

widget_target_name = dbutils.widgets.get("target_name")
widget_timestamp = dbutils.widgets.get("timestamp")
widget_sample_data = dbutils.widgets.get("sample_data")

# COMMAND ----------

# MAGIC %md #### Load inputs

# COMMAND ----------

# MAGIC %md Load sociodemo data

# COMMAND ----------

df_sdm_sociodemo_targets = spark.read.format("delta").load(get_value_from_yaml("paths", "sdm_table_paths", "sdm_sociodemo_targets"))

# COMMAND ----------

# MAGIC %md Load feature store

# COMMAND ----------

# MAGIC %md ####Using feature store loader to load interests

# COMMAND ----------

def read_fs(feature_store):

    return feature_store.select('user_id', 'timestamp').filter(F.col("timestamp") == F.lit("2023-03-10"))
#this reading will be modified 
df = spark.read.format("delta").load(get_value_from_yaml("paths", "feature_store_paths", "user_entity_fs"))
df_fs = read_fs(df)
display(df_fs)

# COMMAND ----------

# MAGIC %md #### Processing

# COMMAND ----------

# only wanted publishers
def process_publishers(df: DataFrame):
    if len(publishers_list) > 0:
        df = (df
              .withColumn('AGE', F.when(F.col('PUBLISHER').isin(publishers_list), F.col('AGE')).otherwise('unknown'))
              .withColumn('GENDER', F.when(F.col('PUBLISHER').isin(publishers_list), F.col('GENDER')).otherwise('unknown'))
             )

    return df

df_process_publishers = process_publishers(df_sdm_sociodemo_targets)
display(df_process_publishers)

# COMMAND ----------

# age categories
def process_age_categories(df: DataFrame):
    return (df
            .withColumn('AGE', F.col('AGE').cast('integer'))
            .withColumn('LABEL_AGE',
                        F.when(F.col('AGE').between(0, 17), '0')
                        .when(F.col('AGE').between(18, 24), '1')
                        .when(F.col('AGE').between(25, 34), '2')
                        .when(F.col('AGE').between(35, 44), '3')
                        .when(F.col('AGE').between(45, 54), '4')
                        .when(F.col('AGE').between(55, 64), '5')
                        .when(F.col('AGE').between(65, 100), '6')
                        .otherwise('unknown')
                      )
            .withColumn('GENDER', F.when(F.col('GENDER') == "1", "0")
                                  .when(F.col('GENDER') == "2", "1")
                       )
           )
    
df_process_age_categories = process_age_categories(df_process_publishers)
display(df_process_age_categories)

# COMMAND ----------

# aggregate per user
def aggregate_targets(df: DataFrame, df_fs: DataFrame):
    df = (df
          .groupBy('USER_ID')
          .agg(
              *[F.array_distinct(F.collect_set(x)).alias(x) for x in ['LABEL_AGE', 'GENDER']],
          )
          .withColumn('LABEL_AGE', F.array_remove(F.col('LABEL_AGE'), 'unknown'))
          .withColumn('GENDER', F.array_remove(F.col('GENDER'), 'unknown'))
          .withColumn('AGE_SIZE', F.size('LABEL_AGE'))
          .withColumn('GENDER_SIZE', F.size('GENDER'))
          # extract target
          .withColumn('sociodemo_targets_age', F.when(F.size('LABEL_AGE')== 1, F.col('LABEL_AGE').getItem(0)).otherwise('unknown'))
          .withColumn('sociodemo_targets_gender', F.when(F.size('GENDER') == 1, F.col('GENDER').getItem(0)).otherwise('unknown'))
         )
    # join to feature store records
    return (df_fs
            .join(df, how='left', on=get_value_from_yaml("featurestorebundle", "entities", "user_entity", "id_column"))
           )
    
df_aggregate_targets = aggregate_targets(df_process_age_categories, df_fs)
display(df_aggregate_targets)

# COMMAND ----------

# MAGIC %md #### Write features

# COMMAND ----------

def features_sociodemo_targets(df: DataFrame, table_name, category_name):
    features_dict = {
        "table":  f"{table_name}",
        "category": f"{category_name}",
        "features":{}
        }
    
    features_dict['features']["sociodemo_targets_age"] = {
        "description": 'Sociodemo target: age',
        "fillna_with": None
    }

    features_dict['features']["sociodemo_targets_gender"] = {
        "description": 'Sociodemo target: gender',
        "fillna_with": None
    }

    return (df
            .select(
                get_value_from_yaml("featurestorebundle", "entities", "user_entity", "id_column"),
                get_value_from_yaml("featurestorebundle", "entity_time_column"),
                'sociodemo_targets_age',
                'sociodemo_targets_gender',
            )
           ), features_dict

df_final = features_sociodemo_targets(df_aggregate_targets, "user", "sociodemo_target_features")[0]
display(df_final)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Metadata

# COMMAND ----------

metadata = features_sociodemo_targets(df_aggregate_targets, "user", "sociodemo_target_features")[1]
display(metadata)
