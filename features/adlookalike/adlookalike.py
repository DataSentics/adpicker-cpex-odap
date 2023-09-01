# Databricks notebook source
# MAGIC %md #### Imports

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.dbutils import DBUtils
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import Normalizer, StandardScalerModel, MinMaxScalerModel, MaxAbsScalerModel, RobustScalerModel
from pyspark.sql.window import Window
import mlflow

import json
from mlflow.tracking import MlflowClient

from src.utils.helper_functions_defined_by_user._functions_ml import ith
from src.utils.helper_functions_defined_by_user._DB_connection_functions import load_mysql_table
from src.utils.helper_functions_defined_by_user.yaml_functions import get_value_from_yaml
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger

client = MlflowClient()
scalers = [Normalizer, StandardScalerModel, MinMaxScalerModel, MaxAbsScalerModel, RobustScalerModel]

# COMMAND ----------

# MAGIC %md Widgets

# COMMAND ----------

dbutils.widgets.dropdown("target_name", "<no target>", ["<no target>"], "01. target name")
dbutils.widgets.text("timestamp", " 2023-08-31", "02. timestamp")
dbutils.widgets.dropdown("sample_data", "complete", ["complete", "sample"], "03. sample data")
dbutils.widgets.dropdown("replace_nulls", "yes",["yes", "no"], "replace nulls by zero")

# COMMAND ----------

widget_target_name = dbutils.widgets.get("target_name")
widget_timestamp = dbutils.widgets.get("timestamp")
widget_sample_data = dbutils.widgets.get("sample_data")
widget_replace_nulls = dbutils.widgets.get("replace_nulls")

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md #### Load feature store

# COMMAND ----------

def get_features_to_load(categories, df):

    df_features = (df.filter(F.col('category').isin(categories)))
    
    return [row.feature for row in df_features.collect()]

categories = ['digital_interest', 'digital_general', 'digital_device']
df_metadata = spark.read.format("delta").load("abfss://gold@cpexstorageblobdev.dfs.core.windows.net/feature_store/metadata/metadata.delta")
list_get_features_to_load = get_features_to_load(categories, df_metadata)
print(list_get_features_to_load)
display(df_metadata)

# COMMAND ----------

# MAGIC %md ####Using feature store loader to load interests

# COMMAND ----------

def read_fs(replace_nulls, fs_date, features_to_load, logger, feature_store):
    df = feature_store.select('user_id', 'timestamp', *features_to_load).filter(F.col("timestamp") == F.lit(F.current_date()))

    # convert string with publishers names to array
    df = df.withColumn("owner_names", F.split(F.col("owner_names_7d"), ","))

    if replace_nulls == 'yes':
        df = df.fillna(0, subset=[c for c in df.columns if c not in [entity.id_column, entity.time_column]])
        logger.info("Application of lookalike models: Missing values of predictors are replaced by zeros.")
    else:
        logger.info("Application of lookalike models: Missing values of predictors are not replaced by zeros.")

    return df


feature_store = spark.read.format("delta").load("abfss://gold@cpexstorageblobdev.dfs.core.windows.net/feature_store/features/user_entity.delta")
df_fs = read_fs(dp.get_widget_value('replace_nulls'), dp.get_widget_value("timestamp"), list_get_features_to_load, root_logger, feature_store)
display(df_fs)

# COMMAND ----------

# MAGIC %md #### Load models from database

# COMMAND ----------

def load_lookalikes_to_score():
    dmp_params = (
        load_mysql_table("lookalike", spark, dbutils)
        .filter(F.col("Model").isNotNull())
    )

    # replace ',' by '_' as df columns cannot contain ','
    dmp_params = dmp_params.withColumn('TP_DMP_id', F.regexp_replace('TP_DMP_id', ',', '_'))

    most_recent_params = dmp_params.groupBy("TP_DMP_id").agg(F.max("id").alias("id"))

    return most_recent_params.join(dmp_params, on=["TP_DMP_id", "id"], how="left")

df_load_lookalikes_to_score = load_lookalikes_to_score()
display(df_load_lookalikes_to_score)

# COMMAND ----------

# MAGIC %md Features

# COMMAND ----------

def get_prediction_cols(lookalike_models_df: DataFrame, table_name, category_name):
    df = lookalike_models_df.toPandas()
    features_dict = {
        "table":  f"{table_name}",
        "category": f"{category_name}",
        "features":{}
        }
    
    for _, row in df.iterrows():
        features_dict['features'][f"lookalike_prob_{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}"] = {
        "description": f"lookalike probability for: {row['TP_DMP_type']}_{row['TP_DMP_id']}, client: {row['client_name']}",
        "fillna_with": None,
        "type": "numerical"
        }

    for _, row in df.iterrows():
        features_dict[f"lookalike_perc_{row['TP_DMP_type']}_{row['TP_DMP_id']}_{row['client_name']}"] = {
        "description": f"lookalike percentile for: {row['TP_DMP_type']}_{row['TP_DMP_id']}, client: {row['client_name']}",
        "fillna_with": -1.0,
        "type": "numerical"
        }


    return features_dict

metadata = get_prediction_cols(df_load_lookalikes_to_score, "user", "lookalike_featues")
display(metadata)

# COMMAND ----------

# MAGIC %md #### Score lookalikes

# COMMAND ----------

def drop_prediction_cols(df: DataFrame, prediction_names):
    return df.drop(*prediction_names)

df_drop_prediction_cols = drop_prediction_cols(df_fs, list(metadata['features'].keys()))

# COMMAND ----------

def lookalikes_scoring(df: DataFrame, df_models: DataFrame, logger):
    # Cookies scoring
    df_models = df_models.toPandas()

    for _, row in df_models.iterrows():
        # extract model information
        model = row["Model"]
        id_dmp_tp = row["TP_DMP_id"]
        type_dmp_tp = row["TP_DMP_type"]
        client_name = row["client_name"]
        target_name = f"lookalike_target_{type_dmp_tp}_{id_dmp_tp}_{client_name}"

        # log information about model
        logger.info(f"Applying lookalike model for {type_dmp_tp}_{id_dmp_tp}, defined as {model}, for client: {client_name}")

        try:
            # extract feature names
            model = json.loads(model)

            # apply model
            model_obj = mlflow.spark.load_model(model["ml_model"])

            for stage in model_obj.stages:
                if type(stage) in scalers:
                    df = df.withColumnRenamed('features', 'features_raw')

                df = stage.transform(df)

            if client_name == 'cpex':
                if target_name in df.columns:
                    not_null_value = (F.col(target_name) != 1)
                else:
                    not_null_value = None
            else:
                if target_name in df.columns:
                    not_null_value = ((F.array_contains(F.col("owner_names"), client_name)) & (F.col(target_name) != 1))
                else:
                    not_null_value = (F.array_contains(F.col("owner_names"), client_name))

            if not_null_value is not None:
                df = (df
                      .filter(not_null_value)
                      .withColumn("score2", ith("probability", F.lit(1)))
                      .withColumn("score_rank2", F.percent_rank().over(Window.partitionBy().orderBy("score2")))
                      .withColumnRenamed("score_rank2", f"lookalike_perc_{type_dmp_tp}_{id_dmp_tp}_{client_name}")
                      .withColumnRenamed("score2", f"lookalike_prob_{type_dmp_tp}_{id_dmp_tp}_{client_name}")
                      .unionByName(df.filter(~(not_null_value)), True)
                      .drop('rawPrediction', 'rawFeatures', 'url_features', 'probability', 'prediction', 'features', 'features_raw')
                     )
            else:
                df = (df
                      .withColumn("score2", ith("probability", F.lit(1)))
                      .withColumn("score_rank2", F.percent_rank().over(Window.partitionBy().orderBy("score2")))
                      .withColumnRenamed("score_rank2", f"lookalike_perc_{type_dmp_tp}_{id_dmp_tp}_{client_name}")
                      .withColumnRenamed("score2", f"lookalike_prob_{type_dmp_tp}_{id_dmp_tp}_{client_name}")
                      .drop('rawPrediction', 'rawFeatures', 'url_features', 'probability', 'prediction', 'features', 'features_raw')
                     )
        except BaseException as e:
            logger.error(f"ERROR: application of model: for {type_dmp_tp}_{id_dmp_tp}, for client: {client_name}, defined as {model}, {e}")
            # null prediction
            df = df.withColumn(f"lookalike_perc_{type_dmp_tp}_{id_dmp_tp}_{client_name}", F.lit(None).cast("double"))
            df = df.withColumn(f"lookalike_prob_{type_dmp_tp}_{id_dmp_tp}_{client_name}", F.lit(None).cast("double"))

    return df

df_lookalikes_scoring =lookalikes_scoring(df_drop_prediction_cols, df_load_lookalikes_to_score, root_logger)
display(df_load_lookalikes_to_score)

# COMMAND ----------

# MAGIC %md #### Write features

# COMMAND ----------

def features_lals_predictions(
    df: DataFrame, features_name
):
    return (df
            .select(
                get_value_from_yaml("featurestorebundle", "entities", "user_entity", "id_column"),
                get_value_from_yaml("featurestorebundle", "entity_time_column"),
                *features_name,
            )
           )
    
df_features_lals_predictons =features_lals_predictions(df_lookalikes_scoring, list(metadata['features'].keys()))
