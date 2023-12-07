# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Sample preparation
# MAGIC
# MAGIC This notebook prepares two delta files with URL SDM sample data. Both files are then saved to `~/interests_monitoring/new_interest/` folder in the ABFS.
# MAGIC
# MAGIC Delta files:
# MAGIC * **url_level_%tag%.delta** - Row represents one user-URL interaction with all tokens and bigrams collected from that URL.
# MAGIC * **user_id_level_%tag%.delta** - Row represents all tokens and bigrams hit by a user (aggregated URL data per user)
# MAGIC
# MAGIC The tag should be used to distinguish different datasets - only a dataset with an already existing tag gets overwritten. This allows to prepare multiple samples (e.g. for different time periods).
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %run ./../../../app/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

# global imports
import daipe as dp

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

import pandas as pd
from datetime import date, datetime, timedelta
from logging import Logger


# project-level imports
from adpickercpex.lib.display_result import display_result

from adpickercpex.solutions._functions_helper import (
    add_binary_flag_widget,
    check_binary_flag_widget, 
    display_list_as_dataframe,
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Widgets & settings

# COMMAND ----------

@dp.notebook_function()
def remove_all_widgets(widgets: dp.Widgets):
    widgets.remove_all()

# COMMAND ----------

@dp.notebook_function()
def create_text_widgets(widgets: dp.Widgets):
    widgets.add_text("end_date", "", "End date")
    widgets.add_text("n_days", "7", "Number of days to include")
    widgets.add_text("n_cookies_max", "", "Max number of cookies")
    widgets.add_text("data_tag", "", "Data sample tag")
    
    add_binary_flag_widget(widgets, name="show_stats")

# COMMAND ----------

@dp.notebook_function("%datalakebundle.table.defaults.storage_name%")
def define_home_dir_path(storage_name):
    return f"abfss://solutions@{storage_name}.dfs.core.windows.net/new_interest_definition/"


@dp.notebook_function()
def define_file_naming():
    return {
        "url_prefix": "url_level_",
        "user_prefix": "user_id_level_",
        "delta_sufix": ".delta/",
    }

# COMMAND ----------

@dp.notebook_function(define_home_dir_path, define_file_naming, dp.get_widget_value("data_tag"))
def define_paths(home_dir_path, file_naming_dict, data_tag):
    return {
        'output_path_url': f'{home_dir_path}{file_naming_dict["url_prefix"]}{data_tag}{file_naming_dict["delta_sufix"]}',
        'output_path_user_id': f'{home_dir_path}{file_naming_dict["user_prefix"]}{data_tag}{file_naming_dict["delta_sufix"]}',
    }

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Parse number of cookies limit:

# COMMAND ----------

@dp.notebook_function(dp.get_widget_value("n_cookies_max"))
def parse_cookies_limit(n_cookies_max, logger: Logger):
    try:
        n_cookies_max_int = int(n_cookies_max)
        logger.info(f"Number of cookies limit: {n_cookies_max_int:,}")
        return n_cookies_max_int
    except ValueError:
        logger.warning("No valid limit set for the number of cookies.")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Load input tables

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Pageview:

# COMMAND ----------

@dp.transformation(
    dp.read_table("silver.sdm_pageview"),
    dp.get_widget_value("end_date"),
    dp.get_widget_value("n_days"),
    display=False,
)
def load_sdm_pageview(df: DataFrame, end_date: str, n_days: str):
    # process end date
    try:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    except:
        end_date = date.today()

    # calculate start date
    start_date = end_date - timedelta(days=int(n_days))

    return (df
            .withColumnRenamed("page_screen_view_date", "DATE")
            .filter((F.col("DATE") >= start_date) & (F.col("DATE") <= end_date))
            .select(F.col("user_id").alias("USER_ID"), 
                    F.col("URL_NORMALIZED"),
                    F.col("DATE"),
                   )
           )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC SDM URL:

# COMMAND ----------

@dp.transformation(dp.read_table("silver.sdm_url"), '%processing.options.use_bigrams%', display=False)
def load_sdm_url(df: DataFrame, use_bigrams: bool):        
    # TODO: columns will be renamed after the main pipeline refactoring
    df_renamed = (df
                  .withColumnRenamed("URL_TOKENS_ALL_CLEANED_UNIQUE", "TOKENS")
                  .withColumnRenamed("URL_TOKENS_ALL_CLEANED_UNIQUE_BIGRAMS", "BIGRAMS")
                 )
    cols = [
        "URL_DOMAIN_1_LEVEL",
        "URL_DOMAIN_2_LEVEL",
        "URL_NORMALIZED",
        "TOKENS",
    ]
    
    if use_bigrams:
        cols.append("BIGRAMS")
    
    return df_renamed.select(*cols)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Create the ouputs 
# MAGIC
# MAGIC On two levels of aggregation (URL / user ID) per unique date.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Combining the input tables (no need for aggregation on the URL level): 

# COMMAND ----------

@dp.transformation(load_sdm_pageview, load_sdm_url, parse_cookies_limit, display=False)
def tokens_url_level(df_pageview: DataFrame, df_url: DataFrame, cookies_limit: int):
    df_join = (df_pageview
               .join(df_url, on="URL_NORMALIZED", how="left")
              )
    if cookies_limit is not None:
        # find what ratio should be sampled  
        keep_ratio = cookies_limit / df_join.count()
        # sample down the original data
        df_join = df_join.sample(fraction=keep_ratio,
                                 withReplacement=False)
        
    return df_join

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Aggregation per user:

# COMMAND ----------

@dp.transformation(tokens_url_level, '%processing.options.use_bigrams%', display=False)
def tokens_user_level(df: DataFrame, use_bigrams: bool):
    cols = ["TOKENS", "BIGRAMS"] if use_bigrams else ["TOKENS"]
    cols_concat_ws = [F.concat_ws(',', col).alias(col) for col in cols]
    cols_collect = [F.concat_ws(',', F.collect_list(F.col(col))).alias(col) for col in cols]
    cols_final = [F.array_remove(F.split(F.col(col), ',').cast('array<string>'), "").alias(col) for col in cols]

    return (df
            .select(
                "USER_ID",
                *cols_concat_ws
            )
            .groupBy("USER_ID")
            .agg(
                F.count(F.lit(1)).alias('count'),
                *cols_collect
            )
            .select(
                "USER_ID",
                *cols_final,
            )
           )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Save outputs
# MAGIC
# MAGIC Saving input on two levels of aggregation.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### URL level 
# MAGIC
# MAGIC (no aggregation needed):

# COMMAND ----------

@dp.transformation(tokens_url_level, display=False)
@dp.delta_overwrite(define_paths.result['output_path_url'], options={'delta.autoOptimize.optimizeWrite': True, 'mergeSchema': True})
def save_url_level(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### User level

# COMMAND ----------

@dp.transformation(tokens_user_level, display=False)
@dp.delta_overwrite(define_paths.result['output_path_user_id'], options={'delta.autoOptimize.optimizeWrite': True, 'mergeSchema': True})
def save_user_level(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## All samples overview
# MAGIC
# MAGIC Info about all available data samples in the home directory.

# COMMAND ----------

@dp.transformation(define_home_dir_path, define_file_naming, display=False)
def build_dataframe_samples(home_dir_path, file_naming_dict, logger: Logger):
    # definition of the prefix lengths & tag-isolation functions 
    len_dict = {k: len(v) for k, v in file_naming_dict.items()}
    isolate_tag_url = lambda x: x[len_dict["url_prefix"]:-len_dict["delta_sufix"]]
    isolate_tag_user = lambda x: x[len_dict["user_prefix"]:-len_dict["delta_sufix"]] 
    
    # get all file names in the home dir
    file_names = [fi.name for fi in dbutils.fs.ls(home_dir_path)]
    # filter the files into 3 disjunct categories (url, user, unknown)
    files_url = [name for name in file_names if name[:len_dict["url_prefix"]] == file_naming_dict["url_prefix"]]
    files_user = [name for name in file_names if name[:len_dict["user_prefix"]] == file_naming_dict["user_prefix"]]
    files_unknown = [name for name in file_names if name not in files_url + files_user]
    if len(files_unknown) > 0:
        logger.warning(f"There are unknown files in the home directory: {files_unknown}")
    
    # isolate the tags
    tags_url = [isolate_tag_url(file_name) for file_name in files_url]
    tags_user = [isolate_tag_user(file_name) for file_name in files_user]
    
    # create a DF with results
    df_tags = pd.DataFrame(index=set(tags_url + tags_user))
    df_tags = (df_tags
               .assign(TAG=df_tags.index,
                       URL_FILE_PRESENT=pd.Series({tag: True for tag in tags_url}),
                       USER_FILE_PRESENT=pd.Series({tag: True for tag in tags_user}))
               .fillna(False)
              )
    df_tags["ALL_PRESENT"] = df_tags["URL_FILE_PRESENT"] & df_tags["USER_FILE_PRESENT"]
    
    return df_tags

# COMMAND ----------

@dp.notebook_function(build_dataframe_samples)
def display_available_samples(df: DataFrame, logger: Logger):
    tags_available = (df
                      .query("ALL_PRESENT")
                      .loc[:, "TAG"]
                      .tolist()
                     )
    
    display_list_as_dataframe(tags_available, name="AVAILABLE TAGS", logger=logger)

# COMMAND ----------

@dp.notebook_function(build_dataframe_samples)
def display_incomplete_samples(df: DataFrame, logger: Logger):
    tags_available = (df
                      .query("not ALL_PRESENT")
                      .loc[:, "TAG"]
                      .tolist()
                     )
    
    display_list_as_dataframe(tags_available, name="INCOMPLETE TAGS", logger=logger)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Show stats

# COMMAND ----------

@dp.transformation(tokens_url_level, dp.get_widget_value("show_stats"), display=False)
@display_result()
def show_stats_url_level(df: DataFrame, show_stats, logger: Logger):
    if not check_binary_flag_widget(show_stats):
        return

    # get the result 
    df_per_date = (df
                   .groupBy("DATE")
                   .count()
                   .orderBy(F.desc("DATE"))
                  )
    # log number of records in total
    n_total = (df_per_date
               .agg(F.sum("count").alias("n_total"))
               .first()["n_total"]
              )
    logger.info(f"Total number of records: {n_total:,}")

    return df_per_date 

# COMMAND ----------

@dp.transformation(tokens_user_level, dp.get_widget_value("show_stats"), display=False)
@display_result()
def show_stats_user_level(df: DataFrame, show_stats, logger: Logger):
    if not check_binary_flag_widget(show_stats):
        return
    
    logger.info(f"Total number of users: {df.count():,}")
