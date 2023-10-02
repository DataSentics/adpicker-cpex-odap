# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Income models - interest score calculation
# MAGIC This notebook serves to calculate interest scores for income models. Each interest is given a score based on marketing surveys (stored in azure storage). These scores are then multiplied by users' interest affinities and these values are summed for each user.  We transform and standardize those values to produce final interest scores for each income bracket. 
# MAGIC
# MAGIC Three income categories are defined: Low (0-25k CZK/mon.), Mid (25k-45k CZK/mon.), High (45k+ CZK/mon.). 

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Imports

# COMMAND ----------


import numpy as np
import pyspark.pandas as ps
import pyspark.sql.functions as F

from src.utils.helper_functions_defined_by_user._abcde_utils import standardize_column_sigmoid

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from scipy.stats import boxcox

from src.schemas.income_schemas import get_income_interest_scores

from src.utils.helper_functions_defined_by_user.yaml_functions import get_value_from_yaml
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger
from src.utils.helper_functions_defined_by_user.table_writing_functions import write_dataframe_to_table

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Config
# MAGIC Configure suffixes for income model brackets and sharpness of sigmoidal standardization function.

# COMMAND ----------

INCOME_MODELS_SUFFIXES = ["low", "mid", "high"]
SHARPNESS = 1

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Initialize widgets and user entity

# COMMAND ----------

dbutils.widgets.text("timestamp", "")

# COMMAND ----------

widget_timestamp = dbutils.widgets.get("timestamp")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Read table with scores of interests

# COMMAND ----------

df_income_interest_coeffs = spark.read.format("delta").load(
    get_value_from_yaml("paths", "income_interest_coeffs")
)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Read interests from FS
# MAGIC Load only list of interest names that have been assigned a non-zero score and fetch them from FS.

# COMMAND ----------

def features_to_load(df):
    lst = df.select("interest").rdd.map(lambda row: row[0]).collect()
    return lst

lst_features_to_load = features_to_load(df_income_interest_coeffs)

# COMMAND ----------

def read_fs(list_features):
    fs_stage1 = (
        spark.read.table("odap_features_user.user_stage1")
        .select("user_id", "timestamp", *list_features)
        .filter(F.col("timestamp") == widget_timestamp)
    )
    return fs_stage1


df_fs = read_fs(lst_features_to_load)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Melt interest affinities to long

# COMMAND ----------

def fs_wide_to_long(df, features):
    df_melted_ps = ps.DataFrame(df).melt(
        id_vars=["user_id", "timestamp"], value_vars=features, var_name="interest"
    )
    return df_melted_ps.to_spark()

df_fs_wide_to_long = fs_wide_to_long(df_fs, lst_features_to_load)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Join interest scores
# MAGIC Join interests with their corresponding score and multiply by interest affinities.

# COMMAND ----------

def add_interest_scores(df_long, df_scores):
    return df_long.join(df_scores, on="interest", how="left").select(
        "user_id",
        "timestamp",
        "interest",
        *[
            (F.col("value") * F.col(f"score_{model}")).alias(f"scaled_score_{model}")
            for model in INCOME_MODELS_SUFFIXES
        ],
    )

df_add_interest_scores = add_interest_scores(df_fs_wide_to_long, df_income_interest_coeffs)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Sum values by user
# MAGIC Sum interest scores mulitplied by interest affinites for each user to create non-standardized interest score for each income category.

# COMMAND ----------

def sum_interest_scores(df):
    return df.groupby("user_id").agg(
        F.max("timestamp").alias("timestamp"),
        *[
            (F.sum(f"scaled_score_{model}").alias(f"interest_score_nonstd_{model}"))
            for model in INCOME_MODELS_SUFFIXES
        ],
    )

df_sum_interest_scores = sum_interest_scores(df_add_interest_scores)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Transform and standardize scores
# MAGIC
# MAGIC Use Box-Cox transform to improve distribution of the score and then standardize using `StandardScaler`.
# MAGIC
# MAGIC The values for Box-Cox transformation are shifted to be positive via a static constant (0.01); the magnitude of the shift should not matter ([see link](https://stats.stackexchange.com/questions/399435/which-constant-to-add-when-applying-box-cox-transformation-to-negative-values)).

# COMMAND ----------

def box_cox_transform(df):
    nonstd_columns = [
        f"interest_score_nonstd_{model}" for model in INCOME_MODELS_SUFFIXES
    ]
    df_pandas = df.select("user_id", "timestamp", *nonstd_columns).toPandas()

    for col in nonstd_columns:
        scores = np.array(df_pandas[col]).reshape(-1)
        scores = (
            scores + np.abs(np.min(scores)) + 0.01
        )  # all values need to be positive for Box-Cox transformation
        fitted = boxcox(scores)
        df_pandas[col] = fitted[0]

    df_spark = spark.createDataFrame(df_pandas)
    return df_spark

df_box_cox_transform = box_cox_transform(df_sum_interest_scores)

# COMMAND ----------

def standard_scaler(df):
    nonstd_columns = [
        f"interest_score_nonstd_{model}" for model in INCOME_MODELS_SUFFIXES
    ]
    std_columns = [f"interest_score_std_{model}" for model in INCOME_MODELS_SUFFIXES]

    vec_ass = VectorAssembler(
        inputCols=nonstd_columns, outputCol="features", handleInvalid="skip"
    )
    df_vec = vec_ass.transform(df)

    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features", withMean=True
    )
    df_scaled = scaler.fit(df_vec).transform(df_vec)

    return df_scaled.withColumn(
        "array", vector_to_array(F.col("scaled_features"))
    ).select(
        "user_id",
        "timestamp",
        *[F.col("array")[i].alias(col) for i, col in enumerate(std_columns)],
    )

df_standard_scalar = standard_scaler(df_box_cox_transform)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Scale scores
# MAGIC Finally use a sigmoid function to squash the interest scores into a (-1, 1) interval.

# COMMAND ----------

def interest_score_final(df):
    return df.select(
        "user_id",
        "timestamp",
        *[
            (standardize_column_sigmoid(f"interest_score_std_{model}", SHARPNESS)).alias(
                f"final_interest_score_{model}"
            )
            for model in INCOME_MODELS_SUFFIXES
        ],
    )

df_final = interest_score_final(df_standard_scalar)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Save final table

# COMMAND ----------

def save_scores(df, logger):
    logger.info(f"Saving {df.count()} rows.")
    return df.withColumn("timestamp", F.to_timestamp("timestamp"))

df_save_scores = save_scores(df_final, root_logger)
schema, info = get_income_interest_scores()

write_dataframe_to_table(
    df_save_scores,
    get_value_from_yaml("paths", "income_interest_scores"),
    schema,
    "overwrite",
    root_logger,
    table_properties=info["table_properties"],
 )
