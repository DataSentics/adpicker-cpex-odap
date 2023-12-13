# Databricks notebook source
# MAGIC %run ./../../app/bootstrap
# COMMAND ----------

import os
import json
from datetime import date, datetime
from logging import Logger

import daipe as dp
from daipecore.widgets.Widgets import Widgets
from pyspark.sql.dataframe import DataFrame

from adpickercpex.utils.mlops.mlflow_tools import retrieve_model, viable_stages

# COMMAND ----------

import logging
logger = logging.getLogger("py4j")
logger.setLevel(logging.ERROR)

logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

# to initialize keys
@dp.notebook_function()
def get_interest_list(fs: dp.fs.FeatureStore):
    lst =  fs.get_metadata()
    return lst

# COMMAND ----------

@dp.notebook_function()
def create_text_widget(widgets: Widgets):
    widgets.add_text("fs_date", "", "FS date") #for manual execution with different date

# COMMAND ----------

@dp.transformation(dp.read_delta("%lookalike.delta_path%"), display=False)
def load_new_lookalikes(lookalike_df):
    df = (lookalike_df
          # .filter("Model IS NULL")
          .select("Model", "TP_DMP_id", "client_name", "TP_DMP_type")
          .dropDuplicates()
          )
    return df

# COMMAND ----------

@dp.notebook_function(load_new_lookalikes, dp.get_widget_value("fs_date"))
def run_model_creation(new_lookalikes: DataFrame, fs_date, logger: Logger):
    if new_lookalikes.count() <= 0:
        return
    
    try:
        fs_date = datetime.strptime(fs_date, "%Y-%m-%d").date()
        logger.info(f"Date set from widget value: {fs_date}")
    except:
        fs_date = date.today()
        logger.info(f"No date set; setting as today: {fs_date}")
    
    for lookalike in new_lookalikes.rdd.map(lambda x: x).collect():
        segment_id = lookalike["TP_DMP_id"]
        client_name = lookalike["client_name"]
        tp_dmp_type = lookalike["TP_DMP_type"]
        train_set_location = f"{os.environ['APP_ENV']}_silver.train_lookalike_{tp_dmp_type}_{segment_id}_{client_name}"
        model_info = lookalike["Model"]
        if model_info is not None:
            model_info = json.loads(model_info)
            model_registry_uri = model_info["mlf_model"]
            _, stage = retrieve_model(model_registry_uri, viable_stages, logger=logger)
            if stage == "Production":
                logger.info(f"Production model found {model_registry_uri}, no retraining triggered ")
                continue


        if not spark.catalog.tableExists(train_set_location):
            logger.info(f"Table {train_set_location} does not exist, it will be created.")
        else:
            logger.info(f"Table {train_set_location} already exists, recreating with latest data.")
        
        train_set_success = dbutils.notebook.run(
                    "./create_train_set",
                    20000,
                    {
                        "fs_date": fs_date,
                        "segment_id": segment_id,
                        "client_name": client_name,
                        "tp_dmp_type": tp_dmp_type
                    }
        )

        # dbutils.notebook.exit(False) used in create_train_set produces string 'False'
        if train_set_success == 'False':
            logger.error(f"Table {train_set_location} creation failed.")
            continue
        else:
            logger.info(f"Table {train_set_location} creation successful.")
        

        if spark.sql(f"select * from {train_set_location}").rdd.isEmpty():
            logger.error(f"Table {train_set_location} exists but it is empty.")
            continue 
        logger.info(f"Starting {segment_id} lookalike model training")

        dbutils.notebook.run(
                    "./create_model",
                    26000, # roughly 7 hours
                    {
                        "train_set_location": train_set_location,
                        "segment_id": segment_id
                    }
        )

