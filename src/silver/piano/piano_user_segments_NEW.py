# Databricks notebook source
# MAGIC %run ./../../app/bootstrap

# COMMAND ----------

import hashlib
import hmac
from datetime import datetime, timedelta

import pyspark.sql.functions as F
import requests
from pyspark.sql.dataframe import DataFrame

import daipe as dp
from adpickercpex.silver.piano.schema_lookalike import get_lookalike_schema
from adpickercpex.silver.piano.schema_piano_user_segments import get_schema_user_segments, get_daipe_schema_user_segments
# pylint: disable = E0401
from src.adpickercpex.solutions._DB_connection_functions import load_mysql_table

# COMMAND ----------

# for key initialization
@dp.transformation()
def get_metadata(fs: dp.fs.FeatureStore):
    df_metadata = fs.get_metadata()
    return df_metadata

# COMMAND ----------

downloaded_interval = 7
start_day = datetime.now() - timedelta(days=downloaded_interval)
start_day_0000 = start_day.replace(hour=0, minute=0, second=0)
end_day_2359 = datetime.now().replace(hour=23, minute=59, second=59)

# COMMAND ----------

SITE_GROUP_ID = "1353717145799506157"
START_TIMESTAMP = int(start_day_0000.timestamp())
END_TIMESTAMP = int(end_day_2359.timestamp())
API_LIMIT = 1000000
PIANO_API_URL = "https://api.cxense.com/traffic/data"

PIANO_USER = dbutils.secrets.get("unit-kv", "piano-name")
PIANO_KEY = dbutils.secrets.get("unit-kv", "piano-pass")

# COMMAND ----------

def get_user_ids(segment_id):
    payload = {
        "siteGroupId": SITE_GROUP_ID,
        'fields'     : [
            'userId'
        ],
        "filters"    : [{"type": "segment", "item": segment_id}],
        'start'      : START_TIMESTAMP,
        'stop'       : END_TIMESTAMP,
        'count'      : API_LIMIT
    }

    date = datetime.utcnow().isoformat() + "Z"
    hmac_obj = hmac.new(PIANO_KEY.encode("utf-8"), date.encode("utf-8"), hashlib.sha256)
    hmac_value = hmac_obj.hexdigest()
    headers = {
        'X-cXense-Authentication': 'username=' + PIANO_USER + ' date=' + date + ' hmac-sha256-hex=' + str(hmac_value)
    }

    response = requests.post(PIANO_API_URL, headers=headers, json=payload)
    body = response.json()
    return body["events"]


# COMMAND ----------

@dp.notebook_function()
@dp.delta_write_ignore("%lookalike.delta_path%")
def create_lookalike_table():
    return spark.createDataFrame(data=spark.sparkContext.emptyRDD(),
                                 schema=get_lookalike_schema())

# COMMAND ----------

@dp.transformation(dp.read_delta("%lookalike.delta_path%"), display=False)
@dp.delta_overwrite("%lookalike.delta_path%")
def update_lookalike_delta(loolalike_df):
    lookalikes_sql_tab = load_mysql_table("lookalike", spark, dbutils)
    new_lal_df = lookalikes_sql_tab.drop("Model").join(loolalike_df.select("Model", "TP_DMP_id"),
                                                       on="TP_DMP_id",
                                                       how="left")
    return new_lal_df

# COMMAND ----------

@dp.transformation(dp.read_delta("%lookalike.delta_path%"), display=False)
def load_segment_ids(lookalike_df):
    return (lookalike_df
            .withColumn("TP_DMP_id", F.explode(F.split("TP_DMP_id", ",")))
            .select("TP_DMP_id")
            .distinct()
            )

# COMMAND ----------

@dp.transformation(load_segment_ids)
def create_user_segment_df(segment_ids_df: DataFrame) -> DataFrame:
    df = spark.createDataFrame([], schema=get_schema_user_segments())
    segment_ids = segment_ids_df.rdd.map(lambda x: x.TP_DMP_id).collect()

    for segment_id in segment_ids:
        user_ids = [(user_id_data["userId"], segment_id)
                    for user_id_data in get_user_ids(segment_id)]  # load users in segment
        segment_df = (spark.createDataFrame(user_ids, schema=get_schema_user_segments())
                      .withColumn('segment_id', F.lit(segment_id))
                      .distinct()
                      )
        df = df.union(segment_df)

    return df

# COMMAND ----------

@dp.transformation(create_user_segment_df)
@dp.table_overwrite("silver.user_segments_piano", table_schema=get_daipe_schema_user_segments())
def save_table(df: DataFrame):
    return df
