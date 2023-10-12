"""
General purpose helper functions that are used in multiple notebooks 
"""

import pandas as pd
import re
from logging import Logger
from typing import Iterable, Callable

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame


def filter_list_by_regex(input_list: list, regex: str):
    """
    Returns only elements which match the regex.
    
    :param input_list: list to be filtered
    :param regex: regular expression used for filtering
    :return: new list with subset of original elements that match the regex
    """
    pattern = re.compile(regex)
    res = [x for x in input_list if pattern.match(x)]
    return res


def str_split_safe(s: str, delim: str, position: int) -> str:
    """
    Retrieves element of a string-split by numeric index.
    str.split wrapper, handles out of range errors by returning None.
    
    :param s: string to be split
    :param delim: delimiter to split the string by
    :param position: index (within the string-split) of the value that should be retrieved 
    :return: element of the string-split on `position` (substring of the original string), None when the index is out of range
    """
    try: 
        return s.split(delim)[position]
    except IndexError: 
        return None
    
    
def get_stats_for_column(df: DataFrame, col: str, n_steps_percentile: int = 10):
    """
    Get average and significant percentiles over given column.
    
    :param df: pyspark DF
    :param col: name of the (numeric) column to collect the stats for
    :param n_steps_percentile: number of percentiles computed (they are uniformly distributed between 0 and 1) 
    """
    # get the stats
    def get_step(i):
        return round(1 / n_steps_percentile * i, 2)

    return (df
            .agg(
                F.mean(col).alias("mean"),
                *[(F.percentile_approx(col, get_step(i)).alias(f"percentile_{get_step(i)}")) 
                  for i in range(1, n_steps_percentile)]
            )
           )
    
    
def display_list_as_dataframe(l: list, name: str = "list", logger: Logger = None) -> pd.DataFrame:
    """
    Uses the list to create a dataframe of 1 column & displays it
    
    :param l: the list to be displayed
    :param name: name of the only column in the dataframe
    :param logger: instance to be used for logging (nothing is logged when no logger is passed)
    :return: created DF
    """
    df = pd.Series(l, name=name, dtype="object").to_frame()
    try:
        df.display()
    except ValueError: 
        if logger is not None:
            logger.info(f"Could not display a dataframe - '{name}': the list is empty.")
        return pd.DataFrame()
    
    return df


def add_binary_flag_widget(widgets, name: str, label: str = None): 
    """
    Adds new widget with "True" / "False" options
    
    :param widgets: daipe widgets object
    :param name: name of the widget
    :param label: label of the widget (when not passed, formatted name is used)
    :return: None
    """
    kwargs_default = {
        "choices": ["True", "False"], 
        "default_value": "False"
    }
    # use a default label deduced from the name if it's not passed
    label = name.replace("_", " ").capitalize() if label is None else label
    
    widgets.add_select(name=name, label=label, **kwargs_default)


def check_binary_flag_widget(flag: str) -> bool: 
    """
    Converts flag (True/False) widgets from string to bool
    
    :param flag: widget value
    :return: logical meaning of the widget content
    """
    return str(flag).lower() == "true"


def get_top_n_sorted(df: DataFrame, column_name: str, n: int) -> DataFrame: 
    """
    Returns only top N rows of a dataframe based on specified column
    
    :param df: spark dataframe to be sorted
    :param column_name: name of the column to sort by
    :param n: number of rows to be returned
    :return: sorted dataframe of 'n' rows
    """
    return (df
            .sort(F.desc(column_name))
            .limit(n)
           )


def flatten_string_iterable(src: Iterable[str], new_collection: Callable = set):
    """
    Flattens any iterable of strings, returning new collection of 
    tokens that are aquired by splitting each of the strings on whitespaces
    """
    return new_collection([str_token
                           for text in src
                           for str_token in text.split()])
    

def format_fraction(num: float, denom: float) -> str: 
    """
    Formats a fraction for printing to also show the percentage.
    
    :param num: numerator of the fraction
    :param denom: denominator of the fraction
    :return: formatted fraction that shows percentage
    """
    return (f"{num:,} / {denom:,} ~ {(float('inf') if denom == 0 else (num / denom)):.2%}")    
       
    
def replace_categorical_levels(df: DataFrame, column_name: str, column_cats: list, replace_with: str = "None"):
    """
    Replaces uninteresting levels of categorical variable with a chosen value.
    
    :param df: dataframe with column to replace
    :param column_name: name of the column in which the levels are replaced
    :param column_cats: list of categorical level names to replace
    :param replace_with: value to replace the unwanted levels
    """
    return df.withColumn(column_name, F.when(F.col(column_name).isin(column_cats), replace_with).otherwise(F.col(column_name)))
    