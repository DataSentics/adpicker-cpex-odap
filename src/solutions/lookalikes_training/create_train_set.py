# Databricks notebook source
# MAGIC %run ./../../app/bootstrap

# COMMAND ----------

import logging
logger = logging.getLogger("py4j")
logger.setLevel(logging.ERROR)

logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

from logging import Logger

# TODO this should not be needed
import daipe as dp
from daipecore.widgets.Widgets import Widgets
from adpickercpex.lib.FeatureStoreTimestampGetter import FeatureStoreTimestampGetter

import pyspark.sql.functions as F
import requests
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.utils import AnalysisException

# COMMAND ----------

@dp.notebook_function("%slack_webhooks.applications%",
                      "AdPicker Reporting",
                      "adpicker-notifications"
                      )
def get_slack_webhook_url(config, application, channel):
    return config[application]["channel_hooks"][channel]

# COMMAND ----------

@dp.notebook_function()
def create_text_widget(widgets: Widgets):
    widgets.add_text("segment_id", "", "segment id")
    widgets.add_text("tp_dmp_type", "", "tp dmp type")
    widgets.add_text("client_name", "", "client name")
    widgets.add_text("fs_date", "", "FS date")  # for manual execution with different date

# COMMAND ----------

USER_COUNT_LIMIT = 500
NEGATIVE_TO_POSITIVE_UNDERSAMPLE_RATIO = 50

# COMMAND ----------

@dp.notebook_function(dp.get_widget_value("segment_id"), 
                      dp.get_widget_value("tp_dmp_type"), 
                      dp.get_widget_value("client_name"))
def init_segment_variables(segment_id, tp_dmp_type, client_name, logger: Logger):
    SEGMENT_ID = dbutils.widgets.get('segment_id').replace(",", "_")
    SEGMENT_COLUMN_NAME = f"lookalike_target_{tp_dmp_type}_{SEGMENT_ID}_{client_name}"
    SEGMENT_TABLE = f"silver.train_lookalike_{tp_dmp_type}_{SEGMENT_ID}_{client_name}"
    logger.info(f"SEGMENT_ID: {SEGMENT_ID}, SEGMENT_COLUMN_NAME: {SEGMENT_COLUMN_NAME}, SEGMENT_TABLE: {SEGMENT_TABLE}")
    return SEGMENT_ID, SEGMENT_COLUMN_NAME, SEGMENT_TABLE

SEGMENT_ID, SEGMENT_COLUMN_NAME, SEGMENT_TABLE = init_segment_variables.result

# COMMAND ----------

# MAGIC %md
# MAGIC Load feature store

# COMMAND ----------

@dp.notebook_function()
def get_interest_list(fs: dp.fs.FeatureStore):
    lst = (
        fs.get_metadata()
        .filter(F.col("category") == "digital_interest")
        .select("feature")
        .collect()
    )
    return [row.feature for row in lst]

# COMMAND ----------

@dp.notebook_function()
def get_features_list(fs: dp.fs.FeatureStore,
                      categories=["digital_interest", "digital_general", "digital_device"]):
    lst = (
        fs.get_metadata()
        .filter(F.col("category").isin(categories))
        .select("feature")
        .collect()
    )
    return [row.feature for row in lst if row.feature]

# COMMAND ----------

user_entity = dp.fs.get_entity()
feature = dp.fs.feature_decorator_factory.create(user_entity)

@dp.notebook_function(get_features_list, user_entity, SEGMENT_COLUMN_NAME, dp.get_widget_value('fs_date'))
def get_labelled_fs(features, entity, segment_column_name, fs_date, fs: FeatureStoreTimestampGetter, logger: Logger):
    try:  # this will happen when SEGMENT_COLUMN_NAME is not yet part of FS (i.e. first day after lookalike was created in db)
        features_with_label = features + [segment_column_name]
        df_fs = (
            fs.get_for_timestamp(entity_name=entity.name, 
                                 timestamp=fs_date, 
                                 features=features_with_label, 
                                 skip_incomplete_rows=True)
              .withColumnRenamed(segment_column_name, "label")
            )
        return df_fs
    except AnalysisException as e:
        logger.error(f"Analysis exception \n\n {e} \n might mean that {segment_column_name=} is not present in FS")
        # exit notebook without raising exception so that init_new_models is not stopped, 
        # note that this produces string 'False' at the calling ntb
        dbutils.notebook.exit(False)

# COMMAND ----------

def send_new_lookalike_alert(webhook_url, segment_id, client_name, message):
    headers = {"Content-Type": "application/json"}
    payload = {
        "blocks" : [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f" :mega: A *new lookalike* has been added for client {client_name} with Segment ID *{segment_id}*. :mega: "
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            }
        ]
    }
    response = requests.post(webhook_url, json=payload, headers=headers)
    return response

# COMMAND ----------

def log_and_send_new_lookalike_alert(webhook_url, segment_id, client_name, message, logger):
    r = send_new_lookalike_alert(webhook_url, segment_id, client_name, message)
    logger.info(message)
    if r.status_code != 200:
        logger.error(f"Slack hook got response: {r.status_code}, {r.reason}")

# COMMAND ----------

@dp.notebook_function(get_labelled_fs, 
                      dp.get_widget_value("segment_id"), 
                      dp.get_widget_value("client_name"),
                      get_slack_webhook_url
                      )
def check_user_count(fs_df, segment_id, client_name, webhook_url, logger: Logger):
    pos_count = fs_df.filter("label == 1").count()
    if pos_count > USER_COUNT_LIMIT:
        message = f"Creating train set with *{pos_count}* samples and training the model... "
        log_and_send_new_lookalike_alert(webhook_url, segment_id, client_name, message, logger)
    else:
        message = f"Target contains only *{pos_count}* users, model creation stopped :exclamation: "
        log_and_send_new_lookalike_alert(webhook_url, segment_id, client_name, message, logger)
        # terminate train set and model creation, but allow init_new_models to continue
        # note that this produces string 'False' at the calling ntb
        dbutils.notebook.exit(False) 

# COMMAND ----------

@dp.transformation(get_labelled_fs)
def undersample(df):
    pos = df.filter(F.col("label") == 1)
    neg = df.filter(F.col("label") == 0)
    total_pos = pos.count()
    total_neg = neg.count()

    if total_pos * NEGATIVE_TO_POSITIVE_UNDERSAMPLE_RATIO < total_neg and total_pos != 0:
        fraction = (total_pos * NEGATIVE_TO_POSITIVE_UNDERSAMPLE_RATIO) / total_neg
        undersampled_df = neg.sample(False, fraction, seed=42).union(pos)
        return undersampled_df
    else:
        return df

# COMMAND ----------

@dp.transformation(undersample)
@dp.table_overwrite(SEGMENT_TABLE)
def save_table(df: DataFrame):
    return df
