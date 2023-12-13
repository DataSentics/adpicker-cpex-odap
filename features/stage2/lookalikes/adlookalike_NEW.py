# Databricks notebook source
# MAGIC %run ../../../app/bootstrap

# COMMAND ----------

# MAGIC %md #### Imports

# COMMAND ----------

import json
from logging import Logger

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.dataframe import DataFrame

import daipe as dp
from adpickercpex.lib.FeatureStoreTimestampGetter import FeatureStoreTimestampGetter
from adpickercpex.solutions._functions_ml import ith
from adpickercpex.solutions.lookalikes_training.preprocessing import interpolate_unknowns
from adpickercpex.utils.mlops.mlflow_tools import retrieve_model, viable_stages


# COMMAND ----------

@dp.notebook_function()
def init_widgets(widgets_factory: dp.fs.WidgetsFactory):
    widgets_factory.create()

# COMMAND ----------

user_entity = dp.fs.get_entity()
feature = dp.fs.feature_decorator_factory.create(user_entity)

# COMMAND ----------

LOOKALIKE_PROBABILITY_PREFIX = "lookalike_prob_"
LOOKALIKE_PERCENTILE_PREFIX = "lookalike_perc_"
LOOKALIKE_FEATURE_PREFIX = "lookalike_target_"

# COMMAND ----------

# MAGIC %md Widgets

# COMMAND ----------

@dp.notebook_function()
def create_text_widget(widgets: dp.Widgets):
    widgets.add_select("replace_nulls", ["yes", "no"], "yes", "replace nulls by zero")

# COMMAND ----------

# MAGIC %md #### Load feature store

# COMMAND ----------

@dp.notebook_function(['digital_interest', 'digital_general', 'digital_device'])
def get_features_to_load(categories, feature_store: dp.fs.FeatureStore):
    df_features = (feature_store
                   .get_metadata()
                   .filter(F.col('category').isin(categories))
                   )

    return [row.feature for row in df_features.collect()]


# COMMAND ----------

# MAGIC %md ####Using feature store loader to load interests

# COMMAND ----------

@dp.transformation(user_entity,
                   dp.get_widget_value('replace_nulls'),
                   dp.get_widget_value("timestamp"),
                   get_features_to_load)
def read_fs(entity,
            replace_nulls,
            fs_date,
            features_to_load,
            feature_store: FeatureStoreTimestampGetter,
            logger: Logger):
    df = (feature_store
          .get_for_timestamp(entity_name=entity.name,
                             timestamp=fs_date,
                             features=features_to_load,
                             skip_incomplete_rows=True)
          )

    # convert string with publishers names to array
    df = df.withColumn("owner_names", F.split(F.col("owner_names_7d"), ","))

    if replace_nulls == 'yes':
        df = df.fillna(0, subset=[c for c in df.columns if c not in [entity.id_column, entity.time_column]])
        logger.info("Application of lookalike models: Missing values of predictors are replaced by zeros.")
    else:
        logger.info("Application of lookalike models: Missing values of predictors are not replaced by zeros.")

    return df


# COMMAND ----------

# MAGIC %md #### Load models from database

# COMMAND ----------

@dp.transformation(dp.read_delta("%lookalike.delta_path%"), display=False)
def load_lookalikes_to_score(lookalike_df):
    # replace ',' by '_' as df columns cannot contain ','
    dmp_params = lookalike_df.withColumn('TP_DMP_id', F.regexp_replace('TP_DMP_id', ',', '_'))

    most_recent_params = dmp_params.groupBy("TP_DMP_id").agg(F.max("id").alias("id"))

    return most_recent_params.join(dmp_params, on=["TP_DMP_id", "id"], how="left")

# COMMAND ----------

# MAGIC %md Features

# COMMAND ----------

@dp.notebook_function(load_lookalikes_to_score)
def get_prediction_cols(lookalike_models_df: DataFrame):
    df = lookalike_models_df.toPandas()
    return {
        'features': [
            dp.fs.Feature(
                f"{LOOKALIKE_PROBABILITY_PREFIX}{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}",
                f"lookalike probability for: {row['TP_DMP_type']}_{row['TP_DMP_id']}, client: {row['client_name']}",
                fillna_with=None,
                type="numerical",
            )
            for _, row in df.iterrows()
        ] + [
            dp.fs.Feature(
                f"{LOOKALIKE_PERCENTILE_PREFIX}{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}",
                f"lookalike percentile for: {row['TP_DMP_type']}_{row['TP_DMP_id']}, client: {row['client_name']}",
                fillna_with=-1.0,
                type="numerical",
            )
            for _, row in df.iterrows()
        ],
        'features_names': [
            name for _, row in df.iterrows() for name in (
                f"{LOOKALIKE_PERCENTILE_PREFIX}{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}",
                f"{LOOKALIKE_PROBABILITY_PREFIX}{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}",
            )
        ]
    }

# COMMAND ----------

@dp.transformation(read_fs, get_prediction_cols.result['features_names'])
def drop_prediction_cols(df: DataFrame, prediction_names):
    return df.drop(*prediction_names)

# COMMAND ----------

# MAGIC %md #### Score lookalikes

# COMMAND ----------

@dp.transformation(drop_prediction_cols, load_lookalikes_to_score)
def score_lookalikes(df, lookalike_models_df, logger: Logger):
    lookalike_models_df = lookalike_models_df.toPandas()

    for _, row in lookalike_models_df.iterrows():
        model = row["Model"]

        percentile_col_name = f"{LOOKALIKE_PERCENTILE_PREFIX}{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}"
        probability_col_name = f"{LOOKALIKE_PROBABILITY_PREFIX}{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}"
        logger.info(f"percentile_col_name: {percentile_col_name}, probability_col_name: {probability_col_name}")

        if model is None:  # model hasn't been trained yet
            df = (df
                  .withColumn(percentile_col_name, F.lit(-1.0))
                  .withColumn(probability_col_name, F.lit(-1.0))
                  )
        else:
            model_info = json.loads(model)
            model_registry_uri = model_info["mlf_model"]
            model_obj, _ = retrieve_model(model_registry_uri, viable_stages, logger=logger)

            original_cols = set(df.columns)
            df = interpolate_unknowns(df)
            df = model_obj.transform(df)
            pipeline_cols = set(df.columns)

            cols_to_drop = list(pipeline_cols - original_cols)

            df = (df
                  .withColumn(probability_col_name, ith("probability", F.lit(1)))
                  .withColumn(percentile_col_name,
                              F.percent_rank()
                              .over(Window.partitionBy()
                                    .orderBy(probability_col_name)))
                  .drop(*cols_to_drop)
                  )
    return df

# COMMAND ----------

# MAGIC %md #### Write features

# COMMAND ----------

@dp.transformation(score_lookalikes, user_entity, get_prediction_cols.result['features_names'])
@feature(
        *get_prediction_cols.result['features'],
        category="lookalike",
)
def features_lals_predictions(
        df: DataFrame, entity, features_names
):
    return (df
    .select(
            entity.id_column,
            entity.time_column,
            *features_names,
    )
    )
