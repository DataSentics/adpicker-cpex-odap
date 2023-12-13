# Databricks notebook source
# MAGIC %run ../../../app/bootstrap

# COMMAND ----------

# MAGIC %md #### Imports

# COMMAND ----------

from logging import Logger

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

import daipe as dp
from adpickercpex.lib.FeatureStoreTimestampGetter import FeatureStoreTimestampGetter

# COMMAND ----------

LOOKALIKE_FEATURE_PREFIX = "lookalike_target_"

# COMMAND ----------

@dp.notebook_function()
def init_widgets(widgets_factory: dp.fs.WidgetsFactory):
    widgets_factory.create()

# COMMAND ----------

user_entity = dp.fs.get_entity()
feature = dp.fs.feature_decorator_factory.create(user_entity)

# COMMAND ----------

# MAGIC %md #### Load inputs

# COMMAND ----------

# MAGIC %md Load lookalike models

# COMMAND ----------

@dp.transformation(dp.read_delta("%lookalike.delta_path%"), display=False)
def load_lookalikes_models(lookalike_df):
    lookalike_models = (
        lookalike_df
        .withColumn("TP_DMP_id", F.regexp_replace("TP_DMP_id", ",", "_"))  #replace ',' by '_' as df column names cannot contain ','
        .select("TP_DMP_id", "client_name", "TP_DMP_type")
        .distinct()
    )
    return lookalike_models

# COMMAND ----------

@dp.notebook_function(load_lookalikes_models)
def get_features(lookalike_models_df: DataFrame):
    df = lookalike_models_df.toPandas()
    return {
        'features': [
            dp.fs.Feature(
                f"{LOOKALIKE_FEATURE_PREFIX}{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}",
                f"lookalike target for: {row['TP_DMP_type']}_{row['TP_DMP_id']}, for client: {row['client_name']}",
                fillna_with=None,
            )
            for _, row in df.iterrows()
        ],
        'features_names': [
            f"{LOOKALIKE_FEATURE_PREFIX}{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}" for _, row in df.iterrows()
        ]
    }

# COMMAND ----------

# MAGIC %md Load feature store

# COMMAND ----------

@dp.transformation(user_entity, dp.get_widget_value("timestamp"))
def read_fs(entity, timestamp, feature_store: FeatureStoreTimestampGetter):
    return (feature_store
            .get_for_timestamp(entity_name=entity.name, timestamp=timestamp, features=[], skip_incomplete_rows=True)
           )

# COMMAND ----------

# MAGIC %md #### Assign target

# COMMAND ----------

@dp.transformation(read_fs, dp.read_table("silver.user_segments_piano"), load_lookalikes_models, user_entity)
def lookalikes_target(df_fs: DataFrame, df_user_segments: DataFrame, df_models: DataFrame, entity, logger: Logger):
    df_models = df_models.toPandas()

    for _, row in df_models.iterrows():
        # extract trait
        lookalike_column_name = f"{LOOKALIKE_FEATURE_PREFIX}{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}"
        segment_id = row["TP_DMP_id"]
        segment_ids_list = segment_id.split("_")

        try:
            # filter observations with given trait
            df_feature = (df_user_segments
                             .filter(F.col("segment_id").isin(segment_ids_list))  
                             # if there are more segments in a lookalike this effectively creates target as their union
                             .withColumn(lookalike_column_name, F.lit(1))
                             .select(entity.id_column, lookalike_column_name)
                             .dropDuplicates()
                            )

            # join to feature store record
            df_fs = (df_fs
                     .join(df_feature, how='left', on=entity.id_column)
                    )
        except BaseException as e:
            logger.error(f"ERROR: adding LaL target for: {segment_id}, {e}")

    return df_fs.fillna(0)

# COMMAND ----------

@dp.transformation(lookalikes_target, get_features.result['features_names'])
def array_of_lals(df: DataFrame, feature_names):
    for feature_name in feature_names:
        df = (df
              .withColumn(f'temp_{feature_name}', 
                          F.when(F.col(feature_name) == 1, feature_name.replace(LOOKALIKE_FEATURE_PREFIX, ''))
                          .otherwise(None))
              )

    return (df
            .withColumn('lookalike_targets', F.concat_ws(',', *[f'temp_{c}' for c in feature_names]))
            .drop(*[f'temp_{c}' for c in feature_names])
           )

# COMMAND ----------

# MAGIC %md #### Write features

# COMMAND ----------

@dp.transformation(array_of_lals, user_entity, get_features.result['features_names'])
@feature(
    *get_features.result['features'],
    dp.fs.Feature('lookalike_targets', 'array of lookalikes for user', fillna_with=None),
    category="lookalike_targets",
)
def features_lals_targets(
    df: DataFrame, entity, features_names
):
    return (df
            .select(
                entity.id_column,
                entity.time_column,
                *features_names,
                'lookalike_targets')
            )
