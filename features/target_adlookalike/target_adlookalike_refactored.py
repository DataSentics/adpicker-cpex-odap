# Databricks notebook source
# MAGIC %md #### Imports

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.dbutils import DBUtils
from pyspark.sql.session import SparkSession

# from adpickercpex.lib.FeatureStoreTimestampGetter import FeatureStoreTimestampGetter

from src.utils.helper_functions_defined_by_user._DB_connection_functions import load_mysql_table
from src.utils.helper_functions_defined_by_user._functions_ml import process_multiple_segments_input
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger
from src.utils.helper_functions_defined_by_user.yaml_functions import get_value_from_yaml

# COMMAND ----------

dbutils.widgets.dropdown("target_name", "<no target>", ["<no target>"], "01. target name")
dbutils.widgets.text("timestamp", "2023-08-03", "02. timestamp")
dbutils.widgets.dropdown("sample_data", "complete", ["complete", "sample"], "03. sample data")

# COMMAND ----------

widget_target_name = dbutils.widgets.get("target_name")
widget_timestamp = dbutils.widgets.get("timestamp")
widget_sample_data = dbutils.widgets.get("sample_data")

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md #### Load inputs

# COMMAND ----------

# MAGIC %md Load table with traits

# COMMAND ----------

df_user_traits = spark.read.format("delta").load("/mnt/aam-cpex-dev/solutions/testing/user_traits.delta")
display(df_user_traits)

# COMMAND ----------

# MAGIC %md Load lookalike models

# COMMAND ----------

def load_lookalikes_models():
    dmp_params = (
        load_mysql_table("lookalike", spark, dbutils)
        .withColumn("next_retraining", F.to_date("next_retraining"))
        .filter(F.col("Model").isNotNull())
    )

    # replace ',' by '_' as df columns cannot contain ','
    dmp_params = dmp_params.withColumn('TP_DMP_id', F.regexp_replace('TP_DMP_id', ',', '_'))

    most_recent_params = dmp_params.groupBy("TP_DMP_id").agg(F.max("id").alias("id"))

    return most_recent_params.join(dmp_params, on=["TP_DMP_id", "id"], how="left")

df_load_lookalikes_models = load_lookalikes_models()
display(df_load_lookalikes_models)

# COMMAND ----------

def get_features(lookalike_models_df: DataFrame, table_name, category_name):
    df = lookalike_models_df.toPandas()
    features_dict = {
        "table":  f"{table_name}",
        "category": f"{category_name}",
        "features":{}
        }
    
    for _, row in df.iterrows():
        features_dict['features'][f"lookalike_target_{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}"] = {
        "description": f"lookalike target for: {row['TP_DMP_type']}_{row['TP_DMP_id']}, for client: {row['client_name']}",
        "fillna_with": None,
        }
        
    return features_dict

metadata = get_features(df_load_lookalikes_models, "user", "lookalike_target_features")



# COMMAND ----------

# MAGIC %md Load feature store

# COMMAND ----------

#TO BE MODIFIED
@dp.transformation(user_entity, dp.get_widget_value("timestamp"))
def read_fs(entity, timestamp, feature_store: FeatureStoreTimestampGetter):
    return (feature_store
            .get_for_timestamp(entity_name=entity.name, timestamp=timestamp, features=[], skip_incomplete_rows=True)
           )

df_fs = 

# COMMAND ----------

# MAGIC %md #### Assign target

# COMMAND ----------

def lookalikes_target(df_fs: DataFrame, df_traits: DataFrame, df_models: DataFrame, logger):
    df_models = df_models.toPandas()

    for _, row in df_models.iterrows():
        # extract trait
        id_dmp_tp = row["TP_DMP_id"]
        type_dmp_tp = row["TP_DMP_type"]
        client_name = row["client_name"]

        id_dmp_tp_list = id_dmp_tp.replace("_", ",")
        id_dmp_tp_list = process_multiple_segments_input(id_dmp_tp_list)['converted_list']

        try:
            # filter observations with given trait
            df_traits_row = (df_traits
                             .filter(F.col('TRAIT').isin(id_dmp_tp_list))
                             .withColumn(f"lookalike_target_{type_dmp_tp}_{id_dmp_tp}_{client_name}", F.lit(1))
                             .select(get_value_from_yaml("featurestorebundle", "entities", "user_entity", "id_column"), f"lookalike_target_{type_dmp_tp}_{id_dmp_tp}_{client_name}")
                             .dropDuplicates()
                            )

            # join to feature store recored
            df_fs = (df_fs
                     .join(df_traits_row, how='left', on=get_value_from_yaml("featurestorebundle", "entities", "user_entity", "id_column"))
                    )
        except BaseException as e:
            logger.error(f"ERROR: adding target of model for: {type_dmp_tp}_{id_dmp_tp}_{client_name}, {e}")
            df = df.withColumn(f"lookalike_target_{type_dmp_tp}_{id_dmp_tp}_{client_name}", F.lit(None).cast('integer'))

    return df_fs.fillna(0)

df_lookalikes_target = lookalikes_target(df_fs, df_user_traits, df_load_lookalikes_models, root_logger)

# COMMAND ----------

@dp.transformation(lookalikes_target, get_features.result['features_names'])
def array_of_lals(df: DataFrame, feature_names):
    for feature_name in feature_names:
        df = df.withColumn(f'temp_{feature_name}', F.when(F.col(feature_name) == 1, feature_name.replace('lookalike_target_', '')).otherwise(None))

    return (df
            .withColumn('lookalike_targets', F.concat_ws(',', *[f'temp_{c}' for c in feature_names]))
            .drop(*[f'temp_{c}' for c in feature_names])
           )
    
df_array_of_lals = array_of_lals(df_lookalikes_target,list(metadata['features'].keys()))

# COMMAND ----------

# MAGIC %md #### Write features

# COMMAND ----------

def features_lals_targets(
    df: DataFrame, features_names
):
    return (df
            .select(
                get_value_from_yaml("featurestorebundle", "entities", "user_entity", "id_column"),
                get_value_from_yaml("featurestorebundle", "entity_time_column"),
                *features_name,
                'lookalike_targets',
            )
           )
    
df_features_lals_targets = features_lals_targets(df_array_of_lals, list(metadata['features'].keys()))
