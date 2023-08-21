# Databricks notebook source
# MAGIC %md
# MAGIC #### Imports and settings

# COMMAND ----------

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from datetime import date, datetime, timedelta
import sys
sys.path.append('../../utils/helper_functions_defined_by_user/')

from schema import get_schema_user_traits, get_schema_user_segments
from table_writing_functions import write_dataframe_to_table
from yaml_functions import get_value_from_yaml

# COMMAND ----------

# MAGIC %md
# MAGIC #### Widgets

# COMMAND ----------

dbutils.widgets.text("end_date", "", "End date")
dbutils.widgets.text("n_days", "7", "Number of days to include")

# COMMAND ----------

# MAGIC %md #### Load silver

# COMMAND ----------

widget_end_date = dbutils.widgets.get("end_date")
widget_n_days = dbutils.widgets.get("n_days")

# COMMAND ----------

def process_dates(end_date, n_days):
    # process end date
    if end_date is not None:
        try:
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        except:
            end_date = date.today()
    else:
        end_date = date.today()

    # calculate start date
    start_date = end_date - timedelta(days=int(n_days))

    return {
        'start_date': start_date,
        'end_date': end_date,
    }

dict_process_dates = process_dates(widget_end_date, widget_n_days)


# COMMAND ----------

def read_silver(
    df: DataFrame,
    start_date,
    end_date,
):
    # to deal for potential minor mismatch between day (from file upload) and DATE (extracted from EVENT_TIME) column
    start_date_safe = start_date - timedelta(days=2)

    return (df
            .filter(F.col("day") >= start_date_safe)
            .select(
                F.col("DEVICE").alias("USER_ID"),
                F.col("EVENT_TIME").alias("TIMESTAMP"),
                F.to_date(F.col("EVENT_TIME")).alias("DATE"),
                'userParameters', #maybe?

            )
            .filter(F.col('DATE') >= start_date).filter(F.col('DATE') <= end_date)
           )
df_bronze_cpex_piano = spark.read.format("delta").load(get_value_from_yaml("paths", "piano_table_paths", "cpex_table_piano"))

df_silver_cpex_piano = read_silver(df_bronze_cpex_piano, dict_process_dates["start_date"], dict_process_dates["end_date"] )


# COMMAND ----------

# MAGIC %md #### calculate traits, segments per device (user_id)

# COMMAND ----------

def aggregate_traits(df: DataFrame, n_days, end_date):
    return (df
            .withColumn('TRAIT', F.explode('userParameters'))
            .groupBy('USER_ID', 'TRAIT')
            .agg(
                F.min(F.col('DATE')).alias('FIRST_DATE')
            )
            .withColumn('RUN_DATE', F.current_date())
            .withColumn('N_DAYS', F.lit(n_days).cast('integer'))
            .withColumn('END_DATE', F.lit(end_date))
           )
df_aggregated_traits = aggregate_traits(df_silver_cpex_piano, widget_n_days, dict_process_dates["end_date"])

# COMMAND ----------

def aggregate_segments(df: DataFrame, n_days, end_date):
    return (df
            .withColumn('SEGMENT', F.explode('userParameters'))
            .groupBy('USER_ID', 'SEGMENT')
            .agg(
                F.min(F.col('DATE')).alias('FIRST_DATE')
            )
            .withColumn('RUN_DATE', F.current_date())
            .withColumn('N_DAYS', F.lit(n_days).cast('integer'))
            .withColumn('END_DATE', F.lit(end_date))
           )
df_aggregated_segments = aggregate_segments(df_silver_cpex_piano, widget_n_days, dict_process_dates["end_date"])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save tables

# COMMAND ----------

def save_user_traits(df: DataFrame):
    return (df
            .select(
                'USER_ID',
                'TRAIT',
                'FIRST_DATE',
                'RUN_DATE',
                'N_DAYS',
                'END_DATE',
            )
           )
    
df_user_traits = save_user_traits(df_aggregated_traits)

schema_user_traits, info_user_traits = get_schema_user_traits()

write_dataframe_to_table(df_user_traits, 
                         get_value_from_yaml("paths", "user_table_paths", "user_traits"), 
                         schema_user_traits, 
                         "overwrite",
                         table_properties=info_user_traits['table_properties'])


# COMMAND ----------

def save_user_segments(df: DataFrame):
    return (df
            .select(
                'USER_ID',
                'SEGMENT',
                'FIRST_DATE',
                'RUN_DATE',
                'N_DAYS',
                'END_DATE',
            )
           )
    
df_user_segments = save_user_segments(df_aggregated_segments)

schema_user_segments, info_user_segments = get_schema_user_segments()

write_dataframe_to_table(df_user_segments, 
                         get_value_from_yaml("paths", "user_table_paths", "user_segments"), 
                         schema_user_segments, 
                         "overwrite",
                         table_properties=info_user_segments['table_properties'])
