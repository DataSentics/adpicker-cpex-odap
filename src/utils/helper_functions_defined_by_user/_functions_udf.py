"""
All general purpose pandas UDFs and utility functions that work with pandas UDFs 
"""

import pandas as pd
import numpy as np
import unidecode

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.dataframe import DataFrame


@F.pandas_udf(T.StringType())
def udf_reverse_word_order(s: pd.Series) -> pd.Series:
    """
    Changes order of words within a string 
    """
    return s.apply(lambda x: " ".join(x.split()[::-1]))


@F.pandas_udf(T.StringType())
def udf_strip_diacritics_str(s: pd.Series) -> pd.Series:
    """
    Replaces special characters with diacritics within a string by basic (stripped) ones
    """
    return s.apply(lambda x: unidecode.unidecode(x))


@F.pandas_udf(T.ArrayType(T.StringType()))
def udf_strip_diacritics_array(s: pd.Series) -> pd.Series:
    """
    Replaces special characters with diacritics within all string elements of an array by basic (stripped) ones
    """
    return s.apply(lambda x: [unidecode.unidecode(s) for s in x])


@F.pandas_udf(T.StringType())
def udf_values_count_str(s: pd.Series) -> pd.Series:
    """
    For an array-type column, computes an absolute frequency of all values 
    and returns the result sorted from the most frequent to the least frequent values.
    
    The result is a joint string of all frequencies separated by a comma,
    with each frequency following a format: <value>: <frequency>
    """
    def _generate_value_counts(arr):
        # get value counts, sort them by frequency (count)
        vals, cnts = np.unique(arr, return_counts=True)
        value_counts = [(val, cnt) for val, cnt in zip(vals, cnts)]
        return sorted(value_counts, key=lambda x: x[1], reverse=True)
    
    return s.apply(lambda x: ", ".join([f"{kw}: {cnt}" 
                                        for kw, cnt 
                                        in _generate_value_counts(x)
                                       ]))

