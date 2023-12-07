# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # Clear monitoring deltas
# MAGIC 
# MAGIC This notebook deletes all data from the output delta tables filled by the interests monitoring job. 

# COMMAND ----------

# MAGIC %run ./../../../app/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Imports

# COMMAND ----------

# global imports
import daipe as dp

from logging import Logger
from collections import namedtuple


# project-level imports
from adpickercpex.solutions._functions_helper import (
    add_binary_flag_widget,
    check_binary_flag_widget,
)

# COMMAND ----------

FileWithPath = namedtuple("FileWithPath", ["file_name", "path"])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Settings

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC All files that can be deleted:

# COMMAND ----------

@dp.notebook_function()
def get_all_delta_files():
    return [
        "interest_keywords_num_hits",
        "interest_keywords_share",
        "affinities_hit_ratio",
        "interest_useful_keywords",
        "affinities_correlation",
        "interest_set_per_keyword",
        "common_keywords_matrix",
        "interest_common_keywords",
    ]

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Widgets
# MAGIC 
# MAGIC Add binary flag (True / False) for each file.

# COMMAND ----------

@dp.notebook_function()
def remove_all_widgets(widgets: dp.Widgets):
    widgets.remove_all()

# COMMAND ----------

@dp.notebook_function(get_all_delta_files)
def create_widgets(delta_files, widgets: dp.Widgets):
    def _make_label(file_name):
        return str(file_name).replace("_", " ")
    # widgets for expected files & add another widget for all other files 
    for file_name in (delta_files + ["unknown_files"]): 
        add_binary_flag_widget(widgets, name=file_name, label=_make_label(file_name))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Path definitions

# COMMAND ----------

@dp.notebook_function('%datalakebundle.table.defaults.storage_name%')
def define_home_dir_path(storage_name):
    return f"abfss://solutions@{storage_name}.dfs.core.windows.net/interests_monitoring/"

# COMMAND ----------

@dp.notebook_function(define_home_dir_path, get_all_delta_files)
def get_all_monitoring_paths(home_dir_path, files): 
    path_factory =  lambda x: f"{home_dir_path}{x}.delta/"
    paths = [path_factory(file_name) for file_name in files]
    
    return paths

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Build paths & check existence: 

# COMMAND ----------

@dp.notebook_function(get_all_monitoring_paths)
def filter_paths(paths, logger: Logger):
    # check presence on by one
    all_ok = True
    for path in paths: 
        try:
            dbutils.fs.ls(path)
        except Exception: 
            all_ok = False
            paths.remove(path)
            logger.warning(f"path '{path}' does not exist.")
    if all_ok: 
        logger.info("All defined paths exist.")
        
    return paths

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Deltas drop

# COMMAND ----------

def _remove_delta(path: str):
    dbutils.fs.rm(path, recurse=True)    

# COMMAND ----------

@dp.notebook_function(get_all_delta_files, get_all_monitoring_paths)
def get_delete_file_paths_delta(files, paths, logger: Logger):
    # only keep files with positive widget values
    files_to_delete = [FileWithPath(file_name=file_name, path=path)
                       for file_name, path in zip(files, paths)
                       if check_binary_flag_widget(dbutils.widgets.get(file_name))]
    
    if len(files_to_delete) > 0:
        logger.warning(f"Files to be deleted: {[f.file_name for f in files_to_delete]}")
    else:
        logger.info("There are no files to be deleted.")
        
    return [f.path for f in files_to_delete]

# COMMAND ----------

@dp.notebook_function(get_delete_file_paths_delta)
def delete_files_delta(paths, logger: Logger):
    for path in paths:
        _remove_delta(path)
        logger.info(f"File @ '{path}' sucesfully dropped.")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## State check

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Directory state after: 

# COMMAND ----------

@dp.notebook_function(define_home_dir_path)
def display_home_dir_state(home_dir, logger: Logger): 
    try:
        display(dbutils.fs.ls(home_dir))
    except ValueError:
        logger.info("There are no files in the home directory.")

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Unexpected files in the directory: 

# COMMAND ----------

@dp.notebook_function(define_home_dir_path, get_all_monitoring_paths)
def get_files_other(home_dir, paths_expected, logger: Logger): 
    files_other = [file_info 
                   for file_info in dbutils.fs.ls(home_dir) 
                   if file_info.path not in paths_expected]
    try:
        display(files_other)
    except ValueError: 
        logger.info("There are no unexppected files.")
        
    return files_other

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Deleting them:

# COMMAND ----------

@dp.notebook_function(define_home_dir_path, get_files_other, dp.get_widget_value("unknown_files"))
def get_delete_file_paths_other(home_dir, dbutils_files, delete_unknown_files, logger: Logger):
    if check_binary_flag_widget(delete_unknown_files):
        # extract all unknown file paths
        paths = [file_info.path for file_info in dbutils_files]
        name_from_path = lambda x: f"./{x[len(home_dir):]}"
        files_to_delete = [FileWithPath(file_name=name_from_path(p), path=p) 
                           for p in paths]
        # drop them
        if len(files_to_delete) > 0:
            logger.warning(f"Files to be removed: {[f.file_name for f in files_to_delete]}")
            
            return [f.path for f in files_to_delete]
        
    logger.info("There are no files to be removed.")
    
    return []

# COMMAND ----------

@dp.notebook_function(get_delete_file_paths_other)
def delete_files_other(paths, logger: Logger):
    for path in paths:
        _remove_delta(path)
        logger.info(f"File @ '{path}' sucesfully dropped.")
