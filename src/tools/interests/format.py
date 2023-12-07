"""
Processing of the raw (delta) interests DF into other formats 
"""

from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F

from collections import namedtuple


# ---------------------------------------- Formatting functions ----------------------------------------


Interest = namedtuple("Interest", ["keywords", "general_interest"])
_add_affinity_prefix = lambda x: f"ad_interest_affinity_{x.lower()}"


def interest_mapping_to_db(df: DataFrame) -> dict: 
    """
    Returns mapping of affinity feature columns to their ID in the MySQL DB
    
    :param df: spark DF with interest definitions
    :return: dictionary in format <interest feature name>: <DB ID>
    """
    return {
        _add_affinity_prefix(row["subinterest"]) : str(row["db_id"])
        for row in df.collect()
    }
    
    
def subinterest_feature_names(df: DataFrame) -> list: 
    """
    Returns names of all interests 
    
    :param df: spark DF with interest definitions
    :return: list of feature names
    """
    return [row["subinterest"].lower()
            for row in df.collect()]
    
    
def affinity_feature_names(df: DataFrame) -> list: 
    """
    Returns names of all features that describe interests' affinities 
    
    :param df: spark DF with interest definitions
    :return: list of feature names
    """
    return [_add_affinity_prefix(row["subinterest"]) 
            for row in df.collect()]
    
    
def interest_definitions_to_dict_single(df: DataFrame) -> dict: 
    """
    Returns all single-word definitions of interests
    
    :param df: spark DF with interest definitions
    :return: dictionary in a format <interest feature name> : <list of keywords>
    """
    return {
        row["subinterest"].lower(): row["keywords"]
        for row in df.collect()
    }
    
    
def interest_definitions_to_dict_bigrams(df: DataFrame) -> dict: 
    """
    Returns all bigram definitions of interests
    
    :param df: spark DF with interest definitions
    :return: dictionary in a format <interest feature name> : <list of keywords>
    """
    return {
        row["subinterest"].lower(): row["keywords_bigrams"]
        for row in df.collect()
    }
    
    
def interest_definitions_to_dict_all(df: DataFrame) -> dict: 
    """
    Returns all definitions of interests
    
    :param df: spark DF with interest definitions
    :return: dictionary in a format <interest feature name> : <list of keywords>
    """
    df_concat = df.withColumn("keywords_all", F.concat(F.col("keywords"), F.col("keywords_bigrams")))
    
    return {
        row["subinterest"].lower(): row["keywords_all"]
        for row in df_concat.collect()
    }


def interest_definitions_to_namedtuple_single(df: DataFrame) -> dict: 
    """
    Returns single-word definitions of interests, along with the general interest that the interest belongs to
    
    :param df: spark DF with interest definitions
    :return: dictionary in a format <interest feature name> : Interest(keywords, general_interest)
    """
    return {
        _add_affinity_prefix(row["subinterest"]):  Interest(
            keywords=row["keywords"], general_interest=row["general_interest"]
        )
        for row in df.collect()
    }


def interest_definitions_to_namedtuple_bigrams(df: DataFrame) -> dict: 
    """
    Returns bigram definitions of interests, along with the general interest that the interest belongs to
    
    :param df: spark DF with interest definitions
    :return: dictionary in a format <interest feature name> : Interest(keywords, general_interest)
    """
    return {
        _add_affinity_prefix(row["subinterest"]):  Interest(
            keywords=row["keywords_bigrams"], general_interest=row["general_interest"]
        )
        for row in df.collect()
    }


def interest_definitions_to_namedtuple_all(df: DataFrame) -> dict: 
    """
    Returns definitions of interests, along with the general interest that the interest belongs to
    
    :param df: spark DF with interest definitions
    :return: dictionary in a format <interest feature name> : Interest(keywords, general_interest)
    """
    df_concat = df.withColumn("keywords_all", F.concat(F.col("keywords"), F.col("keywords_bigrams")))
    
    return {
        _add_affinity_prefix(row["subinterest"]):  Interest(
            keywords=row["keywords_all"], general_interest=row["general_interest"]
        )
        for row in df_concat.collect()
    }





    