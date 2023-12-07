import pandas as pd
from logging import Logger

import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, LongType, FloatType
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.session import SparkSession

from adpickercpex.solutions._functions_helper import (
    str_split_safe,
    display_list_as_dataframe,
    filter_list_by_regex,
)

    
# ------------------------ interest hits ------------------------ 

def _affinities_hit_count(df_affinities: DataFrame, thresholds, spark: SparkSession):
    """
    Calculate number of rows where their affinity exceeds a threshold(s).
    
    :param df_affinities: pyspark DF with `ad_interest_affinity...` columns
    :param thresholds: threshold for which the occurences should be counted
    :return: pyspark dataframe with new `<col>_hit_ratio` columns
    """
    
    # extract all affinity columns
    cols_aff = filter_list_by_regex(df_affinities.columns, regex="^ad_interest_affinity.*")
    # add always-positive flag to indicate records count
    df_affinities = df_affinities.withColumn("all_records", F.lit(1))
    cols_aff.append("all_records")
    # define temp flag columns
    cols_flag = [f"{c}_flag" for c in cols_aff]

    # create empty DF to gradually store counts for all thresholds
    schema = StructType([StructField("threshold", FloatType())] +
                        [StructField(c, LongType()) for c in cols_aff])
    df_result = spark.createDataFrame(data=[], schema=schema)

    # note: iterating through thresholds like this could be slow
    for thresh in thresholds:
        # flag occurences greater than the threshold
        for col_aff, col_flag in zip(cols_aff, cols_flag):
            df_affinities = df_affinities.withColumn(col_flag, F.when(F.col(col_aff) > thresh, 1))
        # count them & append the result
        df_agg = (df_affinities
                  .select(*cols_flag)
                  .agg(*[F.sum(col_flag).alias(col_orig)
                         for col_orig, col_flag in zip(cols_aff, cols_flag)])
                  .withColumn("threshold", F.lit(thresh))
                 )
        df_result = df_result.unionByName(df_agg)

    return df_result


def _affinities_hit_count_to_ratio(df_affinities):
    """
    Translates the hit counts to ratios between 0 and 1.
    
    :param df_affinities: pyspark DF with `ad_interest_affinity...` columns
    :return: pyspark dataframe with new `<col>_hit_ratio` columns
    """
    # compute ratios for all affinity columns
    cols_aff = filter_list_by_regex(df_affinities.columns, regex="^ad_interest_affinity.*")
    for col in cols_aff:
        df_affinities = df_affinities.withColumn(f"{col}_hit_ratio", F.col(col) / F.col("all_records"))

    return df_affinities


def get_affinities_hit_ratio(df_affinities, thresholds, spark: SparkSession):
    """
    Calculates ratio of rows with affinities over threshold(s), for all affinities.
    
    :param df_affinities: pyspark DF with `ad_interest_affinity...` columns
    :param thresholds: list of thresholds for which the occurences should be counted
    :return: pyspark dataframe 1 row per threshold, affinities as columns, hit ratios as values
    """
    # get hit count for each interest
    if "0.0" not in thresholds:
        thresholds.append("0.0")
    df_hit_count = _affinities_hit_count(df_affinities=df_affinities, thresholds=map(float, thresholds), spark=spark)
    # compute & select the ratios
    df_ratio = _affinities_hit_count_to_ratio(df_hit_count)
    return (df_ratio
            .select("threshold", *[c for c in filter_list_by_regex(df_ratio.columns, regex=".*_hit_ratio$")])
            .fillna(0)
           )

    
# ------------------------ affinity correlations ------------------------ 

def get_correlation_between(df_corr, first, second):
    """
    Provided a correlation matrix, returns the correlation between 2 interests.
    
    :param df_corr: pandas DF, correlation matrix
    :param first: first column name
    :param second: second column name
    """
    return df_corr.loc[str(first).upper(), str(second).upper()]


def affinity_correlations_above_threhsold(df_corr, threshold: float = 0.0):
    """
    Returns new pandas DF with list of other columns with correlations over the threshold for each column.
    
    :param df_corr: pandas DF, correlation matrix
    :param threshold: the correlations lower bound
    """
    return df_corr.apply(lambda row: [col_name
                                 for corr_val, col_name in zip(row.drop('index_column'), df_corr.columns)
                                 if corr_val >= threshold and row.index_column != col_name
                                ], axis=1)

    
def affinity_correlations_below_threhsold(df_corr, threshold: float = 1.0):
    """
    Returns new pandas DF with list of other columns with correlations over the threshold for each column.
    
    :param df_corr: pandas DF, correlation matrix
    :param threshold: the correlations upper bound
    """
    series = df_corr.apply(lambda row: [col_name
                                   for corr_val, col_name in zip(row.drop('index_column'), df_corr.columns)
                                   if corr_val <= threshold and row.index_column != col_name
                                  ], axis=1)
    df_corr = series.to_frame(name='correlated')
    df_corr["len_correlated"] = df_corr["correlated"].str.len()
    df_corr.sort_values("len_correlated", axis=0, ascending=False, inplace=True)

    return df_corr


# ------------------------ manual monitoring - selecting ------------------------ 

def get_latest_record(df: DataFrame, logger: Logger = None) -> pd.DataFrame: 
    """
    Returns the latest possible record with `full` run option as a pandas DF.
    This function is specific for loading records from the regular monitoring delta outputs. 
    
    :param df: spark DF with outputs from the regular monitoring
    :param logger: instance to be used for logging (nothing is logged when no logger is passed)
    :return: pandas DF with latest records, None when there are no records with `full` run option
    """
    df_tmp = df.filter(F.col("run_option") == F.lit("full"))
    date_max = df_tmp.agg(F.max("run_date")).first()[0]
    df_filtered = df_tmp.filter(F.col("run_date") == F.lit(date_max)).drop("run_date", "run_option")
    
    if date_max is None:
        if logger is not None:
            logger.warning("There are no records that fulfil the criteria.")
        return
    
    if logger is not None:
        logger.info(f"Latest record run date: {date_max}")
        
    return df_filtered.toPandas()


def get_row_for_interest(df: pd.DataFrame, interest_name: str, logger: Logger = None) -> pd.Series: 
    """
    Returns first row of the DF with the 'interest_name' as index (case insensitive).
    Wrapper for pandas.DataFrame.loc with error handling.
    
    :param df: pandas DF indexed by interest names
    :param interest_name: name of the interest to pick
    :param logger: instance to be used for logging (nothing is logged when no logger is passed)
    :return: first row with given interest as its index, empty row when the interest is not present in the DF index
    """
    try:
        # case insensitive comparison - interest_name should already be parser into lower case
        flags = [interest_name == idx.lower() for idx in df.index]
        return df.loc[flags].iloc[0]
    except IndexError: 
        if logger is not None:
            logger.warning("Unkown interest name.")
    except AttributeError: 
        if logger is not None:
            logger.warning("Cannot load data from the dataframe - it is probably empty.")
        
    return pd.Series(dtype="object")
    

def get_interest_field(row: pd.Series, field_name: str, logger: Logger = None): 
    """
    Returns a field from given interest row.
    Wrapper for pandas.Series.loc with error handling.
    
    :param row: pandas series indexed by the fields
    :param field_name: name of the field (index) to pick
    :param logger: instance to be used for logging (nothing is logged when no logger is passed)
    :return: field identified by given name, None when the field is not present in the series index
    """
    try: 
        return row.loc[field_name]
    except KeyError:
        logger.warning("Unkown field name.")
        return None

    
# ------------------------ manual monitoring - keywords blacklist ------------------------ 

def get_keywords_blacklist_split(keywords_all: list, blacklist: list):
    """
    Takes list of keywords from the definiton and blacklisted keyword as input, returns tuple of 3 new lists: 
    (<keywords that are only in the definition>, 
     <blacklisted keywords from the definition>, 
     <blacklisted keywords that are missing in the definition>)
    
    :param keywords_all: collection of keywords that defines an interest (or its subset)
    :param blacklist: keywords that are to be removed from the interest
    :return: 3 new lists (disjunct) composed of elements of the input lists
    """
    # no conversion to 'set' to keep the ordering for consistency
    # (the inefficiency does not matter on such small collecitons of keywords)
    kws_keep = [kw for kw in keywords_all if kw not in blacklist]
    kws_remove = [kw for kw in blacklist if kw in keywords_all]
    kws_missing = [kw for kw in blacklist if kw not in keywords_all]
    
    return kws_keep, kws_remove, kws_missing


def display_blacklist_split(keywords_all: list, blacklist: list, logger: Logger = None):
    """
    Prints the aftermath of removing blacklisted keywords from a collection. 
    
    :param keywords_all: collection of keywords that defines an interest (or its subset)
    :param blacklist: keywords that are to be removed from the interest
    :param logger: instance to be used for logging (nothing is logged when no logger is passed)
    :return: 3 new lists (disjunct) composed of elements of the input lists 
    """
    kws_keep, kws_remove, kws_missing = get_keywords_blacklist_split(keywords_all, blacklist)
    
    display_list_as_dataframe(kws_keep, name="keywords to stay", logger=logger)
    display_list_as_dataframe(kws_remove, name="keywords to be removed", logger=logger)
    display_list_as_dataframe(kws_missing, name="blacklisted keywords that are missing", logger=logger)
    

# ------------------------ manual monitoring - keyword extraction ------------------------ 

def extract_keyword_into_index(row: pd.Series, data_dtype: str) -> pd.Series: 
    """ 
    Expects a series (with arbitrary index) with values in a format "<keyword>: <data>"
    Returns new series with the <keyword> as its index and <data> as its values.
    All nulls are dropped. 
    
    :param row: series with string "k: v" values
    :param data_dtype: data type of the data series 
    :return: new series indexed with keywords with corresponding data as values
    """
    # only keep values that contain a keyword & data
    row_dropped = row.dropna() 
    # set the keyword as the index, the frequency as the value
    idx = row_dropped.apply(lambda x: str_split_safe(str(x), ":", position=0))
    row_indexed = row_dropped.apply(lambda x: str_split_safe(str(x), " ", position=1)).set_axis(idx.values)
    
    return row_indexed.astype(data_dtype)
    

# ------------------------ manual monitoring - keyword extraction ------------------------ 

def is_row_empty(row: pd.Series, logger: Logger = None) -> pd.Series: 
    """ 
    Pandas series.empty wrapper - checks whether the row is empty or not & allows logging
    
    :param row: any pandas series
    :param logger: logger instance
    :return: same as row.empty
    """
    if row.empty:
        if logger is not None:
            logger.warning("Empty row passed.")
        return True
    return False
    

    