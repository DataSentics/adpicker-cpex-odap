# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # Manual monitoring
# MAGIC 
# MAGIC Interactive analysis of interest definitions using the delta tables saved during the regular interests monitoring process.

# COMMAND ----------

# MAGIC %run ./../../../app/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Imports

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Global imports:

# COMMAND ----------

# global imports
import daipe as dp

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

import pandas as pd
import re
from logging import Logger


# project-level imports
from src.utils.processing_pipelines import process

from src.utils.helper_functions_defined_by_user._functions_helper import (
    display_list_as_dataframe,
    format_fraction,
)

import src.tools.interests.format as interests_format
# TODO this should not be needed
from adpickercpex.lib.display_result import display_result

# local imports
from src.tools.interests.monitoring.utils_interests_monitoring import (
    get_latest_record,
    get_row_for_interest,
    get_interest_field,
    get_keywords_blacklist_split,
    display_blacklist_split,
    extract_keyword_into_index,
    is_row_empty,
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Widgets & settings

# COMMAND ----------

@dp.notebook_function()
def create_widgets(widgets: dp.Widgets):
    # current interest
    widgets.add_text(name="interest_name",
                     default_value="",
                     label="Interest name")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Fill:** list all keywords that should be removed from the definition here:
# MAGIC 
# MAGIC Example: `BLACKLIST_STR = "token_1, token_2, token_3"`

# COMMAND ----------

BLACKLIST_STR = ""

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Parse the settings

# COMMAND ----------

@dp.notebook_function(dp.get_widget_value("interest_name"))
def parse_interest_name(interest_name):
    return interest_name.replace(" ", "_").lower()

# COMMAND ----------

# delimiter of separate values of the 'keywords_blacklist' widget
KWS_BLACKLIST_DELIMITER = ","

@dp.notebook_function()
def parse_keywords_blacklist():
    # parse widget string into a list (ignore the empty element)
    blacklist = [x.strip() for x in BLACKLIST_STR.split(KWS_BLACKLIST_DELIMITER)]
    while "" in blacklist:
        blacklist.remove("")
        
    display_list_as_dataframe(blacklist, name="blacklist")
    
    return blacklist

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Paths
# MAGIC 
# MAGIC Monitoring delta tables paths:

# COMMAND ----------

@dp.notebook_function('%datalakebundle.table.defaults.storage_name%')
def paths_factory(storage_name):
    home_dir_path = f"abfss://solutions@{storage_name}.dfs.core.windows.net/interests_monitoring/"
    
    return lambda x: f"{home_dir_path}{x}.delta"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Read data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Interests definitons:

# COMMAND ----------

@dp.notebook_function('%processing.options.use_stemming%', '%processing.options.use_bigrams%')
def get_processing_strategy(use_stemming: bool, use_bigrams: bool, logger: Logger):
    processing_strategy = process.create_processing_strategy(use_stemming=use_stemming, 
                                                             use_bigrams=use_bigrams, 
                                                             logger=logger)
    return processing_strategy

# COMMAND ----------

@dp.transformation(dp.read_delta('%interests.delta_path%'), get_processing_strategy, display=False)
def read_interests(df: DataFrame, processing_strategy, logger: Logger):
    df_processed = process.process_interest_definitions(df,
                                                        input_col_single="keywords",
                                                        input_col_bigrams="keywords_bigrams",
                                                        processing_strategy=processing_strategy, 
                                                        logger=logger)
    
    return df_processed

# COMMAND ----------

@dp.notebook_function(read_interests, get_processing_strategy)
def get_interest_definitions_to_dict(df: DataFrame, processing_strategy): 
    # pick formatting function based on the use of bigrams
    format_f = (interests_format.interest_definitions_to_dict_all 
                 if process.get_active_flag_bigrams(processing_strategy)
                 else interests_format.interest_definitions_to_dict_single)
    
    return format_f(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Reading the tables:

# COMMAND ----------

@dp.transformation(dp.read_delta(paths_factory.result('interest_keywords_num_hits')), display=False)
def read_interest_keywords_num_hits(df: DataFrame):
    return df

@dp.transformation(dp.read_delta(paths_factory.result('interest_keywords_share')), display=False)
def read_interest_keywords_share(df: DataFrame):
    return df

@dp.transformation(dp.read_delta(paths_factory.result('affinities_hit_ratio')), display=False)
def read_affinities_hit_ratio(df: DataFrame):
    return df

@dp.transformation(dp.read_delta(paths_factory.result('interest_useful_keywords')), display=False)
def read_interest_useful_keywords(df: DataFrame):
    return df

@dp.transformation(dp.read_delta(paths_factory.result('affinities_correlation')), display=False)
def read_affinities_correlation(df: DataFrame):
    return df

@dp.transformation(dp.read_delta(paths_factory.result('interest_set_per_keyword')), display=False)
def read_interest_set_per_keyword(df: DataFrame):
    return df

@dp.transformation(dp.read_delta(paths_factory.result('common_keywords_matrix')), display=False)
def read_common_keywords_matrix(df: DataFrame):
    return df

@dp.transformation(dp.read_delta(paths_factory.result('interest_common_keywords')), display=False)
def read_interest_common_keywords(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Getting the latest record
# MAGIC 
# MAGIC Out of all records in the monitoring tables, getting the latest one with run_option set to `full`.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Number of keyword hits & share: 

# COMMAND ----------

@dp.transformation(read_interest_keywords_num_hits, display=False)
def get_latest_interest_keywords_num_hits(df: DataFrame, logger: Logger): 
    return get_latest_record(df, logger)

# COMMAND ----------

@dp.transformation(read_interest_keywords_share, display=False)
def get_latest_interest_keywords_share(df: DataFrame, logger: Logger): 
    return get_latest_record(df, logger)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Hit ratios: 
# MAGIC 
# MAGIC Aside from picking the latest record possible with `full` run option, filter to `threshold = 0`.

# COMMAND ----------

THRESH = 0.0


@dp.transformation(read_affinities_hit_ratio, display=False)
def get_latest_affinities_hit_ratio(df: DataFrame, logger: Logger): 
    return get_latest_record(df.filter(F.col("threshold") == F.lit(THRESH)), logger)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Keyword usefulness:

# COMMAND ----------

@dp.transformation(read_interest_useful_keywords, display=False)
def get_latest_interest_useful_keywords(df: DataFrame, logger: Logger): 
    df = get_latest_record(df, logger)
    df.index = df.loc[:, "subinterest"]
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Affinity correlations:

# COMMAND ----------

@dp.transformation(read_affinities_correlation, display=False)
def get_latest_affinities_correlation(df: DataFrame, logger: Logger): 
    df = get_latest_record(df, logger)
    df.index = df.loc[:, "index"]
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Keyword uniqueness:

# COMMAND ----------

@dp.transformation(read_interest_set_per_keyword, display=False)
def get_latest_interest_set_per_keyword(df: DataFrame, logger: Logger): 
    df = get_latest_record(df, logger)
    df.index = df.loc[:, "keyword"]

    return df

# COMMAND ----------

@dp.transformation(read_common_keywords_matrix, display=False)
def get_latest_common_keywords_matrix(df: DataFrame, logger: Logger): 
    df = get_latest_record(df, logger)
    df.index = df.loc[:, "subinterest"]
    
    return df

# COMMAND ----------

@dp.transformation(read_interest_common_keywords, display=False)
def get_latest_interest_common_keywords(df: DataFrame, logger: Logger): 
    df = get_latest_record(df, logger)
    df.index = df.loc[:, "subinterest"]
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Static analysis
# MAGIC 
# MAGIC Displaying info about the current interest.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Blacklist info

# COMMAND ----------

@dp.notebook_function(get_interest_definitions_to_dict, parse_interest_name)
def get_all_interest_keywords(interests: dict, interest_name: str, logger: Logger): 
    try:
        return interests[interest_name]
    except KeyError:
        logger.warning(f"There is no such interest: '{interest_name}'")
        return None

# COMMAND ----------

@dp.notebook_function(get_all_interest_keywords, parse_keywords_blacklist)
def display_blacklist_info(keywords_all, blacklist: list, logger: Logger): 
    """
    Displays what words would be removed from the definition & what words would stay.
    Returns a tuple of two lists: (<keywords_to_stay>, <keywords_to_be_removed>)
    """
    if keywords_all is None:
        logger.warning("The keywords definiton is empty.")
        return
    
    kws_keep, kws_remove, kws_missing = get_keywords_blacklist_split(keywords_all, blacklist)
    n_total, n_keep, n_remove, n_missing = map(len, (keywords_all, kws_keep, kws_remove, kws_missing))
    
    if n_missing > 0: 
        logger.warning(f"There are blacklisted keywords ({n_missing})"
                       f" that are not included in the interest definition: {kws_missing}")
              
    logger.info(f"Number of keywords to be kept: {format_fraction(n_keep, n_total)}")
    logger.info(f"Number of keywords to be removed: {format_fraction(n_remove, n_total)}")
    
    display_list_as_dataframe(kws_keep, name="keywords to keep", logger=logger)
    display_list_as_dataframe(kws_remove, name="keyword to remove", logger=logger)
    
    return kws_keep, kws_remove

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Number of keyword hits & share

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Number of hits table:

# COMMAND ----------

@dp.notebook_function(get_latest_interest_keywords_num_hits, parse_interest_name)
def get_row_interest_keywords_num_hits(df: pd.DataFrame, interest_name, logger: Logger):
    df_t = df.transpose()
    return get_row_for_interest(df_t, interest_name, logger)

# COMMAND ----------

@dp.transformation(get_row_interest_keywords_num_hits, parse_keywords_blacklist, display=False)
def display_impact_interest_keywords_num_hits(row: pd.Series, blacklist: list, logger: Logger): 
    if is_row_empty(row, logger):
        return
    row_indexed = extract_keyword_into_index(row, data_dtype="int64")
    row_sorted = row_indexed.sort_values(ascending=False)
    # create a dataframe with number of hits & blacklist flag
    df = pd.DataFrame()
    df["n_hits"] = row_sorted
    df["keyword"] = df.index
    df["blacklisted"] = df["keyword"].apply(lambda x: "Yes" if x in blacklist else "No")
    
    # print the overall impact
    sum_all = df["n_hits"].sum()
    try:
        sum_blacklisted = (df
                           .loc[:, ["blacklisted", "n_hits"]]
                           .groupby("blacklisted").agg(["sum"])
                           .loc["Yes", ("n_hits", "sum")]
                          )
    except KeyError: 
        # no blacklisted keywords
        sum_blacklisted = 0
    logger.info(f"Number of hits of all blacklisted keywords: {format_fraction(sum_blacklisted, sum_all)}")

    df.display()
    
    return df

# COMMAND ----------

@dp.notebook_function(get_latest_interest_keywords_share, parse_interest_name)
def get_row_interest_keywords_share(df: pd.DataFrame, interest_name, logger: Logger):
    df_t = df.transpose()
    return get_row_for_interest(df_t, interest_name, logger)

# COMMAND ----------

@dp.transformation(get_row_interest_keywords_share, parse_keywords_blacklist, display=False)
@display_result()
def display_impact_interest_keywords_share(row: pd.Series, blacklist: list, logger: Logger): 
    if is_row_empty(row, logger):
        return
    row_indexed = extract_keyword_into_index(row, data_dtype="float64")
    row_sorted = row_indexed.sort_values(ascending=False)
    # create a dataframe with number of hits & blacklist flag
    df = pd.DataFrame()
    df["share"] = row_sorted
    df["keyword"] = df.index
    df["blacklisted"] = df["keyword"].apply(lambda x: "Yes" if x in blacklist else "No")
    
    # print the overall impact
    try:
        sum_blacklisted = (df
                           .loc[:, ["blacklisted", "share"]]
                           .groupby("blacklisted").agg(["sum"])
                           .loc["Yes", ("share", "sum")]
                          )
    except KeyError: 
        # no blacklisted keywords
        sum_blacklisted = 0
    logger.info(f"Share of all blacklisted keywords: {sum_blacklisted:.4%}")
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Affinities hit ratios

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Display sorted & formatted:

# COMMAND ----------

@dp.transformation(get_latest_affinities_hit_ratio, display=False)
@display_result()
def display_hit_ratios(df: DataFrame): 
    # rename to drop the prefix & sufix
    name_idx_from, name_idx_to = len("ad_interest_affinity_"), -len("_hit_ratio")
    def _rename_cols(old_col_name):
        return old_col_name[name_idx_from:name_idx_to]
    
    # get sorted DF with one column only with the affinity fields (rename them after filtering)
    affinity_pattern = re.compile("^ad_interest_affinity_")
    df_formatted = (df
                    .iloc[0:1]
                    .rename(lambda x: "hit_ratio")
                    .loc[:, [bool(affinity_pattern.match(idx)) for idx in df.columns]]
                    .rename(mapper=_rename_cols, axis=1)
                    .transpose()
                    .sort_values("hit_ratio", ascending=False)
                   )
    # append interest column
    df_formatted["interest"] = df_formatted.index
    df_formatted["interest"] = df_formatted["interest"].apply(lambda x: x.upper())
    
    return df_formatted

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Usefulness

# COMMAND ----------

@dp.notebook_function(get_latest_interest_useful_keywords, parse_interest_name)
def get_row_interest_useful_keywords(df: pd.DataFrame, interest_name, logger: Logger):
    return get_row_for_interest(df, interest_name, logger)

# COMMAND ----------

@dp.notebook_function(get_row_interest_useful_keywords, parse_keywords_blacklist)
def display_keywords_useless(row: pd.Series, blacklist: list, logger: Logger): 
    if is_row_empty(row, logger):
        return
    kws_useless = get_interest_field(row, "keywords_useless", logger)
    display_blacklist_split(kws_useless, blacklist, logger)
    
    return kws_useless

# COMMAND ----------

@dp.notebook_function(get_row_interest_useful_keywords, parse_keywords_blacklist)
def display_keywords_useful(row: pd.Series, blacklist: list, logger: Logger): 
    if is_row_empty(row, logger):
        return
    kws_useful = get_interest_field(row, "keywords_useful", logger)
    display_blacklist_split(kws_useful, blacklist, logger)
    
    return kws_useful

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Correlation

# COMMAND ----------

@dp.notebook_function(get_latest_affinities_correlation, parse_interest_name)
def get_row_affinities_correlation(df: pd.DataFrame, interest_name, logger: Logger):
    return get_row_for_interest(df, interest_name, logger)

# COMMAND ----------

@dp.transformation(get_row_affinities_correlation, display=False)
@display_result()
def display_affinity_correlations(row: pd.Series, logger: Logger): 
    if is_row_empty(row, logger):
        return
    # create a single-column DF from the fetched row 
    row_dropped = row.drop("index")
    df = row_dropped.astype("float").to_frame(name="correlation")
    # append interest column
    df["interest_name"] = row_dropped.index
    df_sorted = df.sort_values("correlation", ascending=False)
    
    return df_sorted

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Common keywords

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Common keywords per other interest:

# COMMAND ----------

@dp.notebook_function(get_latest_common_keywords_matrix, parse_interest_name)
def get_row_common_keywords_matrix(df: pd.DataFrame, interest_name, logger: Logger):
    return get_row_for_interest(df, interest_name, logger)

# COMMAND ----------

@dp.transformation(get_row_common_keywords_matrix, get_all_interest_keywords, display=False)
@display_result()
def display_common_keywords_matrix(row: pd.Series, all_keywords, logger: Logger):
    if is_row_empty(row, logger):
        return 
    # create a single-column DF from the fetched row 
    row_dropped = row.drop("subinterest")
    df = row_dropped.to_frame(name="common_keywords")
    # calculate both absolute and relative size of overlaps
    df["n_common"] = df["common_keywords"].apply(lambda x: 0 if x is None else len(x))
    size_total = 0 if all_keywords is None else int(len(all_keywords))
    df["ratio_common"] = df["n_common"].apply(lambda x: 0 if size_total == 0 else x / size_total)
    # append interest column
    df["interest"] = row_dropped.index
    # sort by the relative size
    df_sorted = df.sort_values("ratio_common", ascending=False)
    
    return df_sorted

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Uniqueness

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Overall unique / common:

# COMMAND ----------

@dp.notebook_function(get_latest_interest_common_keywords, parse_interest_name)
def get_row_interest_common_keywords(df: pd.DataFrame, interest_name, logger: Logger):
    return get_row_for_interest(df, interest_name, logger)

# COMMAND ----------

@dp.notebook_function(get_row_interest_common_keywords, parse_keywords_blacklist)
def display_keywords_common(row: pd.Series, blacklist: list, logger: Logger): 
    if is_row_empty(row, logger):
        return
    kws_common = get_interest_field(row, "common_keywords", logger)
    display_blacklist_split(kws_common, blacklist, logger)
    
    return kws_common

# COMMAND ----------

@dp.notebook_function(get_row_interest_common_keywords, parse_keywords_blacklist)
def display_keywords_unique(row: pd.Series, blacklist: list, logger: Logger): 
    if is_row_empty(row, logger):
        return
    kws_unique = get_interest_field(row, "keywords_unique", logger)
    display_blacklist_split(kws_unique, blacklist, logger)
    
    return kws_unique

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Interactive part
# MAGIC 
# MAGIC (manual part)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### API definition
# MAGIC 
# MAGIC Definition of callable functions that display particular information from the output of the transformations above.

# COMMAND ----------

@dp.notebook_function()
def get_daipe_logger(logger: Logger):
    return logger


LOGGER_DP = get_daipe_logger.result

# COMMAND ----------

def _retrieve_safe(df: pd.DataFrame, idx, col, on_empty): 
    """
    Returns value of DF defined by index & column if it both exists, value of 'on_empty' otherwise
    """
    if df is None:
        LOGGER_DP.warning(f"Cannot retrive data from the DF (it is None). Returning default value insted: {on_empty}")
        return on_empty
    try: 
        return df.loc[idx, col]
    except KeyError:
        LOGGER_DP.warning(f"Value not found at [{idx}, {col}]. Returning default value insted: {on_empty}")
        return on_empty

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Keyword API:

# COMMAND ----------

def get_num_hits(keyword: str) -> int: 
    """
    Returns number of hits of given keyword
    """
    return _retrieve_safe(display_impact_interest_keywords_num_hits.result, idx=keyword, col="n_hits", on_empty=0)


def get_share(keyword: str) -> float: 
    """
    Returns share of hits of given keyword
    """
    return _retrieve_safe(display_impact_interest_keywords_share.result, idx=keyword, col="share", on_empty=0.0)


def get_usefulness(keyword: str) -> str: 
    """
    Returns a string that represents whether the keyword is useful for current interest or not
    """
    if keyword in display_keywords_useful.result:
        return "USEFUL"
    if keyword in display_keywords_useless.result:
        return "USELESS"
    
    return "UNKNOWN"
    

def get_commonness(keyword: str) -> str:
    """
    Returns a string that represents whether the keyword is unique for current interest or not
    """
    if keyword in display_keywords_common.result:
        return "COMMON"
    if keyword in display_keywords_unique.result:
        return "UNIQUE"
    
    return "UNKNOWN"
      

def get_all_interests(keyword: str, display_as_dataframe=True) -> list: 
    """
    Returns all interests that include given keyword in its definition
    """
    interests = list(_retrieve_safe(get_latest_interest_set_per_keyword.result, idx=keyword, col="subinterests", on_empty=[]))
    if display_as_dataframe:
        display_list_as_dataframe(interests, "interests")
        
    return interests

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Interest API:

# COMMAND ----------

def get_hit_ratio(interest_name: str) -> float:
    """
    Returns hit ratio for given interest (expects format with undescores)
    """
    return _retrieve_safe(display_hit_ratios.result, idx=interest_name.lower(), col="hit_ratio", on_empty=0.0)


def get_correlation(interest_name: str) -> float:
    """
    Returns correlation between given interest and the current one (expects format with undescores)
    """
    return _retrieve_safe(display_affinity_correlations.result, idx=interest_name.upper(), col="correlation", on_empty=0.0)


def get_common_keywords(interest_name: str, display: bool = True) -> list:
    """
    Returns list of all keywords that given interest has with the current one (expects format with undescores)
    """
    keywords_common = list(_retrieve_safe(display_common_keywords_matrix.result, idx=interest_name.upper(), col="common_keywords", on_empty=[]))
    if display:
        display_list_as_dataframe(keywords_common, "common keywords")
    return keywords_common

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Playground
# MAGIC 
# MAGIC Space for other queries:

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


