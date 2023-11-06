# Databricks notebook source
import hashlib
import hmac
from datetime import datetime, timedelta

import pyspark.sql.functions as F
import requests
from pyspark.sql.dataframe import DataFrame

from src.utils.helper_functions_defined_by_user.table_writing_functions import (
    write_dataframe_to_table,
    delta_table_exists,
)
from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger

from src.schemas.piano_segments_schema import get_piano_segments_schema
from src.schemas.lookalike_schema import get_lookalike_schema

# pylint: disable = E0401
from src.utils.helper_functions_defined_by_user._DB_connection_functions import (
    load_mysql_table,
)

# COMMAND ----------

LOOKALIKE_PATH = get_value_from_yaml("paths", "lookalike_path")

# COMMAND ----------

root_logger = instantiate_logger()

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
        "fields": ["userId"],
        "filters": [{"type": "segment", "item": segment_id}],
        "start": START_TIMESTAMP,
        "stop": END_TIMESTAMP,
        "count": API_LIMIT,
    }

    date = datetime.utcnow().isoformat() + "Z"
    hmac_obj = hmac.new(PIANO_KEY.encode("utf-8"), date.encode("utf-8"), hashlib.sha256)
    hmac_value = hmac_obj.hexdigest()
    headers = {
        "X-cXense-Authentication": "username="
        + PIANO_USER
        + " date="
        + date
        + " hmac-sha256-hex="
        + str(hmac_value)
    }

    response = requests.post(PIANO_API_URL, headers=headers, json=payload)
    body = response.json()
    return body["events"]


# COMMAND ----------


def create_lookalike_table(logger):
    if not delta_table_exists(LOOKALIKE_PATH):
        schema, info = get_lookalike_schema()
        df_empty = spark.createDataFrame([], schema)

        write_dataframe_to_table(
            df_empty,
            get_value_from_yaml("paths", "lookalike_path"),
            schema,
            "default",
            logger,
        )


create_lookalike_table(root_logger)

# COMMAND ----------


def update_lookalike_delta(logger):
    lookalike_df = spark.read.format("delta").load(LOOKALIKE_PATH)

    lookalikes_sql_tab = load_mysql_table("lookalike", spark, dbutils)
    new_lal_df = lookalikes_sql_tab.drop("Model").join(
        lookalike_df.select("Model", "TP_DMP_id"), on="TP_DMP_id", how="left"
    )

    schema, info = get_lookalike_schema()

    write_dataframe_to_table(
        new_lal_df,
        get_value_from_yaml("paths", "lookalike_path"),
        schema,
        "overwrite",
        logger,
        table_properties=info["table_properties"],
    )


update_lookalike_delta(root_logger)

# COMMAND ----------


def load_segment_ids():
    lookalike_df = spark.read.format("delta").load(LOOKALIKE_PATH)

    return (
        lookalike_df.withColumn("TP_DMP_id", F.explode(F.split("TP_DMP_id", ",")))
        .select("TP_DMP_id")
        .distinct()
    )


segment_ids_df = load_segment_ids()

# COMMAND ----------


def create_user_segment_df(segment_ids_df: DataFrame) -> DataFrame:
    user_segments_schema, info = get_piano_segments_schema()

    df = spark.createDataFrame([], schema=user_segments_schema)
    segment_ids = segment_ids_df.rdd.map(lambda x: x.TP_DMP_id).collect()

    for segment_id in segment_ids:
        user_ids = [
            (user_id_data["userId"], segment_id)
            for user_id_data in get_user_ids(segment_id)
        ]  # load users in segment

        segment_df = (
            spark.createDataFrame(user_ids, schema=user_segments_schema)
            .withColumn("segment_id", F.lit(segment_id))
            .distinct()
        )
        df = df.union(segment_df)

    return df


user_segments_df = create_user_segment_df(segment_ids_df)

# COMMAND ----------


def save_table(df: DataFrame, logger):
    schema, info = get_piano_segments_schema()

    write_dataframe_to_table(
        df,
        get_value_from_yaml("paths", "user_segments_path"),
        schema,
        "overwrite",
        logger,
        partition=info["partition_by"],
        table_properties=info["table_properties"],
    )


save_table(user_segments_df, root_logger)
