from typing import List, Dict

from pyspark.sql import functions as F

WEB_FEATURES = ["web_analytics_device_type_most_common_7d",
                "web_analytics_device_os_most_common_7d",
                "web_analytics_device_browser_most_common_7d",
                "web_analytics_page_search_engine_most_common_7d",
                "web_analytics_visit_time_most_common_7d",
                "web_analytics_time_on_site_avg_7d",
                "web_analytics_web_security_affinity_7d"]

CATEGORICAL = ["web_analytics_device_type_most_common_7d",
               "web_analytics_device_os_most_common_7d",
               "web_analytics_device_browser_most_common_7d",
               "web_analytics_page_search_engine_most_common_7d",
               "web_analytics_visit_time_most_common_7d",
               "sociodemo_targets_age",
               "sociodemo_targets_gender"]

IGNORED_COLS = ["user_id", "timestamp", "owner_names_7d",
                "web_analytics_device_browser_last_used_7d",
                "web_analytics_page_search_engine_last_used_7d",
                "web_analytics_device_os_last_used_7d",
                "web_analytics_device_type_last_used_7d",
                "web_analytics_blog_days_since_last_visit_7d",  # may contain NULLs
                "lookalike_targets"]

ALLOWED_VALUES = {"web_analytics_device_type_most_common_7d"       : ["mobile", "desktop"],
                  "web_analytics_device_os_most_common_7d"         : ["android", "windows", "ios", "macos", "linux"],
                  "web_analytics_device_browser_most_common_7d"    : ["chrome", "safari", "edge", "mozilla"],
                  "web_analytics_page_search_engine_most_common_7d": ["google", "seznam", "centrum"]}


def interpolate_unknowns(df, column_known_values: Dict[str, List[str]]=None):
    """ Replace unknown values in given columns with literal 'unknown'.

    Params:
        df: dataframe
        column_known_values: dict with columns as keys and lists of known values as values,
                    default is ALLOWED_VALUES
    """
    if column_known_values is None:
        column_known_values = ALLOWED_VALUES
    for key, value in column_known_values.items():
        df = df.withColumn(key, F.when(F.col(key).isin(value),
                                       F.col(key))
                           .otherwise(F.lit("unknown")))
    return df


def get_relevant_colnames(fs):

    feature_colnames = [elem for elem in fs.columns if elem not in ["label", "user_id", "timestamp"]]
    numerical_colnames = [elem for elem in feature_colnames if elem not in CATEGORICAL]
    categorical_colnames = [elem for elem in feature_colnames if elem in CATEGORICAL]
    cat_cols_indexed = [f"{colname}_indexed" for colname in categorical_colnames]
    cat_cols_encoded = [f"{colname}_encoded" for colname in categorical_colnames]
    return numerical_colnames, categorical_colnames, cat_cols_indexed, cat_cols_encoded
