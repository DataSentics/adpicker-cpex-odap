from pyspark.sql import DataFrame, Column
from pyspark.sql import functions as F

sdm_hash = F.xxhash64


def with_columns_renamed(df: DataFrame, old_names: list, new_names: list):
    """Return DataFrame with names of columns renamed according to the given list of new_names."""
    return df.select(*old_names).toDF(*new_names)


def get_max(df: DataFrame, col_to_get_max_from: Column):
    """Return maximum timestamp of already processed data for incremental purposes."""
    return df.agg(F.max(F.col(col_to_get_max_from))).collect()[0][0]
