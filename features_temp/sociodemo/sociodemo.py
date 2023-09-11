# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Sociodemo models application
# MAGIC This notebook serves to apply the trained sociodemo ML models to the data and write the probabilities/percentiles features into the FS. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Imports

# COMMAND ----------

import mlflow
import logging
import pyspark.sql.functions as F
from logging import Logger
from pyspark.sql.dataframe import DataFrame
# pylint: disable=W0614
# pylint: disable=W0401
from pyspark.sql.types import * 
from pyspark.sql.window import Window

from src.utils.helper_functions_defined_by_user._functions_ml import ith
from src.utils.helper_functions_defined_by_user._functions_helper import replace_categorical_levels
from src.utils.helper_functions_defined_by_user.yaml_functions import get_value_from_yaml
# pylint: disable=W0621
# pylint: disable=W0123
# pylint: disable=W1514
# pylint: disable=W0614

# COMMAND ----------

logger = logging.getLogger("py4j")
logger.setLevel(logging.ERROR)

logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

PREFIX_LIST = ["men", "women", *[f"cat{_index}" for _index in range(7)]]

SEARCH_ENGINE_VALUES_TO_REPLACE = ["ask", "maxask", "volny", "poprask"]
DEVICE_OS_VALUES_TO_REPLACE = ["ipados", "ios"]
DEVICE_TYPE_VALUES_TO_REPLACE = ["TV"]

ALLOWED_VALUES = {"web_analytics_device_type_most_common_7d": ["mobile", "desktop"],
"web_analytics_device_os_most_common_7d": ["android", "windows", "ios", "macos", "linux"],
"web_analytics_device_browser_most_common_7d": ["chrome", "safari", "edge", "mozilla"],
"web_analytics_page_search_engine_most_common_7d": ["google", "seznam", "centrum"]}

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Initialization

# COMMAND ----------

dbutils.widgets.dropdown("target_name", "<no target>", ["<no target>"], "01. target name")
dbutils.widgets.text("timestamp", "2020-12-12", "02. timestamp")
dbutils.widgets.dropdown("sample_data", "complete", ["complete", "sample"], "03. sample data")

# COMMAND ----------

widget_target_name = dbutils.widgets.get("target_name")
widget_timestamp = dbutils.widgets.get("timestamp")
widget_sample_data = dbutils.widgets.get("sample_data")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Get features & data
# MAGIC Feature names and data from FS is fetched; the collected URLs for each user are then joined.

# COMMAND ----------

def get_features_to_load(categories):
    df_features = spark.read.table("odap_features_user.metadata").filter(
        F.col("category").isin(categories)
    )
    return [row.feature for row in df_features.collect()]

categories = ["digital_interest", "digital_general", "digital_device", "sociodemo_targets"]
list_features_to_load = get_features_to_load(categories)
print(list_features_to_load)

# COMMAND ----------

def read_fs(fs_date,features_to_load):
    fs_stage1 = (
        spark.read.table("odap_features_user.user_stage1")
        .select("user_id", "timestamp", *features_to_load)
        .filter(F.col("timestamp") == widget_timestamp))
    return fs_stage1

df_fs = read_fs(widget_timestamp, list_features_to_load)
display(df_fs)

# COMMAND ----------

def join_urls(read_fs, df_url, logger: Logger):
    df_joined = read_fs.join(df_url.select("user_id", "timestamp", "collected_urls"),on=["user_id", "timestamp"], how="left")
    
    logger.info(f"Count of users with no URLs is {df_joined.select(F.sum(F.col('collected_urls').isNull().cast('integer'))).collect()[0][0]}.")

    return df_joined

df_income_url_scores = spark.read.format("delta").load(get_value_from_yaml("paths", "income_table_paths", "income_url_scores"))
df_join_urls = join_urls(df_fs, df_income_url_scores, logger)
display(df_join_urls)

# COMMAND ----------

def get_cols_from_schema(schema):
    unique_urls = []
    columns_list = ["collected_urls"]
    num_cols = []
    cat_cols = []

    for f in schema:

        if f.name.split("_")[-1] == "flag":
            unique_urls.append((".").join(f.name.split("_")[:-1]))
            num_cols.append(f.name)

        else:
            columns_list.append(f.name)
            if isinstance(f.dataType, StringType):
                cat_cols.append(f.name)
            elif isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType)):
                num_cols.append(f.name)
            else:
                raise Exception(f"{f.name} is unknown type {f.dataType}.") 
    return unique_urls, columns_list, num_cols, cat_cols

# COMMAND ----------

def get_schemas(models_dict):
    with open(f'src/adpickercpex/solutions/sociodemo_training/schemas/socdemo_gender_schema_{models_dict["gender_male"].split("/")[-3]}.txt', 'r') as f:
        schema_gender = eval(f.readlines()[0])


    with open(f'src/adpickercpex/solutions/sociodemo_training/schemas/socdemo_age_schema_{models_dict["ageGroup_0_17"].split("/")[-3]}.txt', 'r') as f:
        schema_age =  eval(f.readlines()[0])

    schema_gender.extend(schema_age)
    schema_both = list(set(schema_gender))
    schema_both.remove(StructField('label', StringType(), True))

    return schema_both

models_dict = get_value_from_yaml("paths", "models", "sociodemo")
schema = get_schemas(models_dict)
print(schema)

# COMMAND ----------

unique_urls, columns_list, num_cols, cat_cols = get_cols_from_schema(schema)
columns_list.extend(["user_id", "timestamp"])

# COMMAND ----------

def choose_features(df, columns_list):
    df = (df.select(*columns_list,*[(F.array_contains(F.col("collected_urls"), domain).cast("int")).alias(f"{domain.replace('.', '_')}_flag") for domain in unique_urls])
            .drop('collected_urls'))
    return df

df_choose_features = choose_features(df_join_urls, columns_list)
display(df_choose_features)

# COMMAND ----------

def replace_rare_values(df, num_cols, cat_cols):
    for key, value in ALLOWED_VALUES.items():
        df = df.withColumn(key, F.when(F.col(key).isin(value), F.col(key)).otherwise("None"))  

    df = replace_categorical_levels(
        df,
        column_name="web_analytics_page_search_engine_most_common_7d",
        column_cats=SEARCH_ENGINE_VALUES_TO_REPLACE,
    )

    df = replace_categorical_levels(
        df,
        column_name="web_analytics_device_os_most_common_7d",
        column_cats=DEVICE_OS_VALUES_TO_REPLACE,
    )

    df = replace_categorical_levels(
        df,
        column_name="web_analytics_device_type_most_common_7d",
        column_cats=DEVICE_TYPE_VALUES_TO_REPLACE,
    )

    df = df.fillna(0, subset=num_cols)
    df = df.fillna("None", subset=cat_cols)
    
    return df

df_replace_rare_values = replace_rare_values(df_choose_features, num_cols, cat_cols)
display(df_replace_rare_values)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Define socdemo features

# COMMAND ----------

def get_features(models_dict, table_name, category_name):
    features_dict = {
        "table":  f"{table_name}",
        "category": f"{category_name}",
        "features":{}
        }

    models_list = list(models_dict.keys()) + ['gender_male']

    for model_name in models_list:
        features_dict['features'][f"sociodemo_perc_{model_name}"] = {
        "description": f"sociodemo percentile for: {model_name}",
        "fillna_with": -1.0
    }
    for model_name in models_list:
        features_dict['features'][f"sociodemo_prob_{model_name}"] = {
        "description": f"sociodemo probability for: {model_name}",
        "fillna_with": None
    }
    return features_dict

models_dict = get_value_from_yaml("paths", "models", "sociodemo")
metadata = get_features(get_value_from_yaml("paths", "models", "sociodemo"), "user", "sociodemo_features")


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Apply sociodemo models

# COMMAND ----------

def apply_model(df, models_dict, logger: Logger):

    for model in models_dict:

        logger.info(f"Applying sociodemo model {model}, with path {models_dict[model]}")

        try:
            model_obj = mlflow.spark.load_model(models_dict[model])
            df = model_obj.transform(df)

            # drop columns created by the ML Pipeline
            to_drop_col = (
                [c for c in df.columns if c.endswith("_indexed")]
                + [c for c in df.columns if c.endswith("_encoded")]
                + [
                    "rawPrediction",
                    "probability",
                    "prediction",
                    "features",
                    "num_features",
                    "num_features_scaled",
                ]
            )
            if "num_feat_raw" in df.columns:
                to_drop_col.append("num_feat_raw")
            if "num_feat_norm" in df.columns:
                to_drop_col.append("num_feat_raw")

            df = (
                df.withColumn("score2", ith("probability", F.lit(1)))
                .withColumn(
                    "score_rank2",
                    F.percent_rank().over(Window.partitionBy().orderBy("score2")),
                )
                .withColumnRenamed("score_rank2", f"sociodemo_perc_{model}")
                .withColumnRenamed("score2", f"sociodemo_prob_{model}")
                .drop(*to_drop_col)
            )
        except BaseException as e:
            logger.error(f"ERROR: application of model: {model}, {e}")
            # null prediction
            df = df.withColumn(f"sociodemo_perc_{model}", F.lit(None).cast("double"))
            df = df.withColumn(f"sociodemo_prob_{model}", F.lit(None).cast("double"))


    # define values for male model as a complement to the female model
    df = df.withColumn(
        "sociodemo_perc_gender_female", 1 - F.col("sociodemo_perc_gender_male")
    )
    df = df.withColumn(
        "sociodemo_prob_gender_female", 1 - F.col("sociodemo_prob_gender_male")
    )

    return df


df_apply_model = apply_model(df_replace_rare_values, models_dict, logger)
display(df_apply_model)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Write features

# COMMAND ----------

def features_sociodemo(
    df: DataFrame, features_names
):

    return df.select(
        "user_id",
        "timestamp,
        *features_name,
    )
df_final = features_sociodemo(df_apply_model, list(metadata['features'].keys()))

# COMMAND ----------

metadata =  {
    "table":  "user_stage2",
    "category": "sociodemo_features",
    "features": {
        "sociodemo_perc_{model_name}": {
            "description": "sociodemo percentile for: {model_name}",
            "fillna_with": -1.0,
        },
        "sociodemo_prob_{model_name}": {
            "description": "sociodemo probability for: {model_name}",
            "fillna_with": None,
        },
    }
    }
