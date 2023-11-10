import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from typing import Optional, List
from datetime import datetime

spark = SparkSession.builder.appName("MyApp").getOrCreate()


def is_valid_timestamp(timestamp):
    try:
        datetime.strptime(timestamp, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def fetch_fs_stage(
    timestamp: str, stage: int, feature_list: Optional[List[str]] = None
) -> DataFrame:
    """
    Fetch a chosen Feature Store stage for a given timestamp.

    Parameters:
        timestamp (str): Timestamp for which to fetch the Feature Store stage.
        stage (int): Stage of the Feature Store to be fetched.
        feature_list (list, optional): The list of features to be fetched. If blank, fetches all features.

    Returns:
        DataFrame: Spark DataFrame with the Feature Store stage.
    """

    # TODO - rewrite not to use dbfs, but ADLS cloud storage, if desired and needed

    if not is_valid_timestamp(timestamp):
        raise BaseException("Invalid timestamp!")

    try:
        fs = spark.read.table(f"odap_features_user.user_stage{stage}").filter(
            F.col("timestamp") == timestamp
        )
    except BaseException as e:
        print(f"An error occurred while loading the FS: {e}.")
        return None

    if feature_list is None:
        return fs
    try:
        fs_selected_features = fs.select("user_id", "timestamp", *feature_list)
        return fs_selected_features
    except BaseException as e:
        print(f"An error occurred while selecting features: {e}.")
        return None
