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
import pyspark.sql.functions as F
from logging import Logger
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import types as T
from pyspark.sql.window import Window
from typing import List

# from adpickercpex.lib.FeatureStoreTimestampGetter import FeatureStoreTimestampGetter
from src.utils.helper_functions_defined_by_user import URL_lists 
from src.utils.helper_functions_defined_by_user._functions_ml import ith
from src.utils.helper_functions_defined_by_user._functions_helper import replace_categorical_levels
from src.utils.helper_functions_defined_by_user.yaml_functions import get_value_from_yaml
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Config
# MAGIC `ALL_URL_LISTS` is a collected list of all domain URLs of interest (for individual gender and age categories).
# MAGIC
# MAGIC `PREFIX_LIST` is a list of strings with URL feature prefixes for easier feature creation. 
# MAGIC
# MAGIC `ALL_URL_LISTS_CLEANED` and `ALL_URLS_CLEANED` are lists of cleaned URL domains (periods in URL replaced by underscores) for better column naming.
# MAGIC
# MAGIC The remaining lists are lists of rarely occuring levels of categorical features with low significance - they are replaced with the `None` category before fitting to reduce the number of (rather useless) coefficients.

# COMMAND ----------

def _clean_url_list(url_list):
    return [domain.replace(".", "_") for domain in url_list]

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

ALL_URL_LISTS = [
    URL_lists.men_primary_url,
    URL_lists.men_secondary_url,
    URL_lists.women_primary_url,
    URL_lists.women_secondary_url,
    URL_lists.cat0_primary_url,
    URL_lists.cat0_secondary_url,
    URL_lists.cat1_primary_url,
    URL_lists.cat1_secondary_url,
    URL_lists.cat2_primary_url,
    URL_lists.cat2_secondary_url,
    URL_lists.cat3_primary_url,
    URL_lists.cat3_secondary_url,
    URL_lists.cat4_primary_url,
    URL_lists.cat4_secondary_url,
    URL_lists.cat5_primary_url,
    URL_lists.cat5_secondary_url,
    URL_lists.cat6_primary_url,
    URL_lists.cat6_secondary_url,
]

PREFIX_LIST = ["men", "women", *[f"cat{_index}" for _index in range(7)]]

ALL_URL_LISTS_CLEANED = [_clean_url_list(lst) for lst in ALL_URL_LISTS]
ALL_URLS_CLEANED = set([domain for lst in ALL_URL_LISTS_CLEANED for domain in lst])

SEARCH_ENGINE_VALUES_TO_REPLACE = ["ask", "maxask", "volny", "poprask"]
DEVICE_OS_VALUES_TO_REPLACE = ["ipados", "ios"]
DEVICE_TYPE_VALUES_TO_REPLACE = ["TV"]

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Initialization

# COMMAND ----------

dbutils.widgets.dropdown("target_name", "<no target>", ["<no target>"], "01. target name")
dbutils.widgets.text("timestamp", "2023-08-03", "02. timestamp")
dbutils.widgets.dropdown("sample_data", "complete", ["complete", "sample"], "03. sample data")

# COMMAND ----------

widget_target_name = dbutils.widgets.get("target_name")
widget_timestamp = dbutils.widgets.get("timestamp")
widget_sample_data = dbutils.widgets.get("sample_data")

# COMMAND ----------

df_fs_metadata = spark.read.format("delta").load("abfss://gold@cpexstorageblobdev.dfs.core.windows.net/feature_store/metadata/metadata.delta")
display(df_fs_metadata)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Get features & data
# MAGIC Feature names and data from FS is fetched; the collected URLs for each user are then joined.

# COMMAND ----------

def get_features_to_load(categories: List[str], df: DataFrame) -> List[str]:

    df_features = df.filter(F.col("category").isin(categories))

    return [row.feature for row in df_features.collect()]

list_categories = ["digital_interest", "digital_general", "digital_device"]
df_features_to_load = get_features_to_load(list_categories, df_fs_metadata) 


# COMMAND ----------

#TO BE MODIFIED
@dp.transformation(user_entity, dp.get_widget_value("timestamp"), get_features_to_load)
def read_fs(
    entity,
    fs_date,
    features_to_load,
    feature_store: FeatureStoreTimestampGetter,  
):
    df = feature_store.get_for_timestamp(
        entity_name=entity.name,
        timestamp=fs_date,
        features=features_to_load,
        skip_incomplete_rows=True,
    )
    return df
df_fs = 

# COMMAND ----------

def join_urls(df_fs, df_url, logger):
    df_joined = df_fs.join(
        df_url.select("user_id", "timestamp", "collected_urls"),
        on=["user_id", "timestamp"],
        how="left",
    )
    logger.info(
        f"Count of users with no URLs is {df_joined.select(F.sum(F.col('collected_urls').isNull().cast('integer'))).collect()[0][0]}."
    )
    return df_joined

df_income_url_scores = spark.read.format("delta").load(get_value_from_yaml("paths", "income_table_paths", "income_url_scores"))
df_join_urls = join_urls(df_fs,  df_income_url_scores, root_logger)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Create URL flags
# MAGIC We create binary integer flags for all URLs of interest.

# COMMAND ----------

def create_domain_features(df):
    df = df.select(
        "*",
        *[
            (F.array_contains("collected_urls", domain.replace("_", ".")).cast("int")).alias(f"{domain}_flag")
            for domain in ALL_URLS_CLEANED
        ]
    )
    return df

df_create_domin_features = create_domain_features(df_join_urls)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC Specific columns are created for each gender and age group category to reduce the number of columns in the model (concentrate more info into less columns). 

# COMMAND ----------

def create_url_flags(df):
    # define specific flagging function
    def _add_category_url_features(df, _index):
        df = df.select(
            "*",
            F.greatest(
                *[F.col(f"{domain}_flag") for domain in ALL_URL_LISTS_CLEANED[2 * _index]]
            ).cast("string").alias(f"{PREFIX_LIST[_index]}_primary_flag"),
            F.greatest(
                *[F.col(f"{domain}_flag") for domain in ALL_URL_LISTS_CLEANED[2 * _index + 1]]
            ).cast("string").alias(f"{PREFIX_LIST[_index]}_secondary_flag"),
        )
        return df
    
    for _index in range(len(PREFIX_LIST)):
        df = _add_category_url_features(df, _index)
    
    return df.drop("collected_urls", *[f"{domain}_flag" for domain in ALL_URLS_CLEANED])

df_create_url_flags = create_url_flags(df_create_domin_features)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Prepare data for model fitting
# MAGIC Replace null values and rarely occuring categorical variable levels.

# COMMAND ----------


def get_data_features(df):
    skip_cols = [get_value_from_yaml("featurestorebundle", "entities", "user_entity", "id_column"), get_value_from_yaml("featurestorebundle", "entity_time_column")]
    cat_cols = [
        f.name
        for f in df.schema.fields
        if isinstance(f.dataType, T.StringType) and f.name not in skip_cols
    ]
    num_cols = [
        f.name
        for f in df.schema.fields
        if isinstance(
            f.dataType, (T.IntegerType, T.DoubleType, T.FloatType, T.LongType)
        )
        and f.name not in skip_cols
    ]
    
    return {
        "skip_cols" : skip_cols,
        "num_cols" : num_cols,
        "cat_cols" : cat_cols 
    }

df_get_data_features = get_data_features(df_create_url_flags)

# COMMAND ----------

def replace_rare_values(df, num_cols, cat_cols):

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

df_replace_rare_values = replace_rare_values(df_create_url_flags, df_get_data_features["num_cols"],  df_get_data_features["cat_cols"])

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Define socdemo features

# COMMAND ----------

#TO BE MODIFIED
@dp.notebook_function("%models.sociodemo%")
def get_features(models_dict):
    models_list = list(models_dict.keys()) + ['gender_male']

    features_tupple = (
            feature_definition for model_name in models_list for feature_definition in (
                (f"sociodemo_perc_{model_name}", f"sociodemo percentile for: {model_name}", -1.0),
                (f"sociodemo_prob_{model_name}", f"sociodemo probability for: {model_name}", None),
            )
    )

    return {
        "features": [
            dp.fs.Feature(
                name,
                description,
                fillna_with=na_val,
                type="numerical",
            )
            for name, description, na_val in features_tupple
        ],
        "names": [
            f"sociodemo_perc_{model_name}" for model_name in models_list
        ] + [
            f"sociodemo_prob_{model_name}" for model_name in models_list
        ]
    }

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Apply sociodemo models

# COMMAND ----------

#TO BE MODIFIED
@dp.transformation(replace_rare_values, "%models.sociodemo%")
def apply_model(df: DataFrame, models_dict, logger: Logger):
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
        "sociodemo_perc_gender_male", 1 - F.col("sociodemo_perc_gender_female")
    )
    df = df.withColumn(
        "sociodemo_prob_gender_male", 1 - F.col("sociodemo_prob_gender_female")
    )
    return df

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Write features

# COMMAND ----------

#TO BE MODIFIED
@dp.transformation(apply_model, get_features.result["names"], user_entity)
@feature(
    *get_features.result["features"],
    category="sociodemo",
)
def features_sociodemo(
    df: DataFrame, features_names, entity
):

    return df.select(
        entity.id_column,
        entity.time_column,
        *features_names,
    )
