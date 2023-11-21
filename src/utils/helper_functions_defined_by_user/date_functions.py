import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame


def get_max_date(df: DataFrame, date_col: str, datetime_col: str, last_n_days=20):
    """
    Get the maximum datetime value from a DataFrame within the last N days.

    This function calculates the maximum datetime value from the specified DataFrame within
    the last N days, based on the provided date and datetime columns.

    Parameters:
        df (DataFrame): The DataFrame from which to calculate the maximum datetime value.
        date_col (str): The name of the column containing the date values for filtering.
        datetime_col (str): The name of the column containing the datetime values for aggregation.
        last_n_days (int, optional): The number of days to consider for the calculation.
            Defaults to 20.

    Returns:
        datetime.datetime: The maximum datetime value within the specified date range.
    """
    return (
        df.filter(
            F.col(date_col) >= F.date_add(F.current_date(), -last_n_days)
        )  # just last n days (to avoid loading whole table)
        .agg(F.max(datetime_col))
        .collect()[0][0]
    )
