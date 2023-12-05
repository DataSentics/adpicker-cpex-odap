# Databricks notebook source
from logging import Logger

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.window import Window

from src.schemas.sdm_schemas import (
    get_schema_sdm_preprocessed,
    get_schema_sdm_session,
    get_schema_sdm_pageview,
)
from src.utils.helper_functions_defined_by_user._functions_nlp import (
    df_url_normalization,
)
from src.utils.helper_functions_defined_by_user.date_functions import get_max_date
from src.utils.helper_functions_defined_by_user.logger import instantiate_logger
from src.utils.helper_functions_defined_by_user.sdm_table_functions import sdm_hash
from src.utils.helper_functions_defined_by_user.table_writing_functions import (
    write_dataframe_to_table,
    delta_table_exists,
)
from src.utils.read_config import config

# COMMAND ----------

root_logger = instantiate_logger()

# COMMAND ----------

tablet_names = ["ipad", "lnv_tab", "galaxy_tab"]
device_brand_names = [
    "apple",
    "nokia",
    "huawei",
    "samsung",
    "xiaomi",
    "aligator",
    "motorola",
    "lenovo",
    "lg",
    "google",
]
device_marketing_names = [
    "iphone",
    "mac",
    "ipad",
    "lnv_tab",
    "galaxy_tab",
    "galaxy",
    "velvet",
    "smart_tv",
    "P30_lite",
    "redmi",
    "mi",
    "pixel",
]
device_os = ["ios", "ipados", "android", "windows", "macos", "linux"]
desktop_os = ["windows", "macos", "linux"]
web_browsers = ["mozilla", "safari", "edge", "opera", "seznam", "chrome"]

# COMMAND ----------

# MAGIC %md Create empty tables

# COMMAND ----------

if not delta_table_exists(
    config.paths.sdm_session
):

    schema, info = get_schema_sdm_session()
    df_empty = spark.createDataFrame([], schema)

    write_dataframe_to_table(
        df_empty,
        config.paths.sdm_session,
        schema,
        "default",
        root_logger,
        info["partition_by"],
        info["table_properties"],
    )

# COMMAND ----------

if not delta_table_exists(
    config.paths.sdm_pageview
):

    schema, info = get_schema_sdm_pageview()
    df_empty = spark.createDataFrame([], schema)

    write_dataframe_to_table(
        df_empty,
        config.paths.sdm_pageview,
        schema,
        "default",
        root_logger,
        info["partition_by"],
        info["table_properties"],
    )

# COMMAND ----------


@F.pandas_udf("string")
def vectorized_udf(REFERER: pd.Series) -> pd.Series:
    def helper_func(ref_string):
        search_engine = ""
        if not ref_string:
            return None
        ref = ref_string.lower()
        if "google" in ref:
            search_engine = "google"
        elif "seznam" in ref:
            search_engine = "seznam"
        elif "bing" in ref:
            search_engine = "bing"
        elif "centrum" in ref:
            search_engine = "centrum"
        elif "yahoo" in ref:
            search_engine = "yahoo"
        elif "volny" in ref:
            search_engine = "volny"
        elif "ask" in ref:
            search_engine = "ask"
        else:
            search_engine = None
        return search_engine

    return REFERER.apply(helper_func)


# COMMAND ----------

# MAGIC %md Load cleansed table

# COMMAND ----------


def read_cleansed_data(df: DataFrame, df_pageview: DataFrame, logger: Logger):
    # get max date in table
    destination_max_date = get_max_date(
        df=df_pageview,
        date_col="page_screen_view_date",
        datetime_col="page_screen_view_timestamp",
        last_n_days=20,
    )

    logger.info(
        f"maximal date in silver.sdm_session before append: {destination_max_date}"
    )

    if destination_max_date is not None:
        df = df.filter(
            F.col("day") >= F.to_date(F.lit(destination_max_date.date()))
        ).filter(F.col("EVENT_TIME") > destination_max_date)
    df = df.withColumnRenamed("EVENT_TIME", "TIMESTAMP")
    df = df.withColumn("DATE", F.to_date(F.col("TIMESTAMP")))
    df = df.withColumn("USER_AGENT", F.lower(F.col("userAgent")))
    df = df.withColumn("IP_ADDRESS", F.lit(None).cast(T.StringType()))

    df = df.withColumn("referrer_search_engine", vectorized_udf(F.col("referrerHost")))
    df = df.withColumn("pageurl_search_engine", vectorized_udf(F.col("rp_pageurl")))
    df = df.withColumn(
        "SEARCH_ENGINE", F.coalesce(df.referrer_search_engine, df.pageurl_search_engine)
    )

    df = df.withColumnRenamed("referrerUrl", "REFERRAL_PATH_FULL")
    df = df.withColumnRenamed("referrerHost", "REFERER")

    df = df.withColumnRenamed("rp_pageurl", "URL")

    return df


# reading path must be changed with the final path
df_bronze_cpex_piano = spark.read.format("delta").load(
    config.paths.cpex_table_piano
)
df_silver_sdm_pageview = spark.read.format("delta").load(
    config.paths.sdm_pageview
)

df_cleansed_data = read_cleansed_data(
    df_bronze_cpex_piano, df_silver_sdm_pageview, root_logger
)

# COMMAND ----------

# df_cleansed_data.printSchema()
# df_graph = df_cleansed_data.filter(F.col("DEVICE") == "").groupby("DATE").count().toPandas().sort_values(by="DATE")
# print(df_cleansed_data.count())
# print(df_cleansed_data.filter(F.col("DEVICE") == "").count())

# import seaborn as sns


# sns.set(rc={"figure.figsize": (11.7, 8.27)})
# g = sns.lineplot(data=df_graph, x="DATE", y="count")

# display(df_graph)

# COMMAND ----------


def filter_empty_ids(df):
    df_filtered = df.filter(F.col("DEVICE") != "")
    return df_filtered


df_filtered_data = filter_empty_ids(df_cleansed_data)

# COMMAND ----------


def calculate_device_features(df: DataFrame):
    return (
        df
        # more detailed info about device
        .withColumn(
            "DEVICE_BRAND_NAME",
            F.when(
                (F.col("USER_AGENT").contains("iphone"))
                | (F.col("USER_AGENT").contains("macintosh"))
                | (F.col("USER_AGENT").contains("ipad")),
                "apple",
            )
            .when(F.col("USER_AGENT").contains("nokia"), "nokia")
            .when(
                (F.col("USER_AGENT").contains("huawei"))
                | (F.col("USER_AGENT").contains("MARX-L")),
                "huawei",
            )
            .when(
                (F.col("USER_AGENT").contains("samsung"))
                | (F.col("USER_AGENT").contains("sm-t")),
                "samsung",
            )
            .when(
                F.col("USER_AGENT").contains("redmi")
                | (F.col("USER_AGENT").contains("mi")),
                "xiaomi",
            )
            .when(F.col("USER_AGENT").contains("aligator"), "aligator")
            .when(F.col("USER_AGENT").contains("moto"), "motorola")
            .when(F.col("USER_AGENT").contains("lenovo"), "lenovo")
            .when(F.col("USER_AGENT").contains("lm-g900"), "lg")
            .when(F.col("USER_AGENT").contains("pixel"), "google"),
        )
        .withColumn(
            "DEVICE_MARKETING_NAME",
            F.when(F.col("USER_AGENT").contains("iphone"), "iphone")
            .when(F.col("USER_AGENT").contains("macintosh"), "mac")
            .when(F.col("USER_AGENT").contains("ipad"), "ipad")
            .when(
                (F.col("USER_AGENT").contains("lenovo tab"))
                | (F.col("USER_AGENT").contains("lenovo tb")),
                "lnv_tab",
            )
            .when(F.col("USER_AGENT").contains("sm-t"), "galaxy_tab")
            .when(F.col("USER_AGENT").contains("sm-g"), "galaxy")
            .when(F.col("USER_AGENT").contains("lm-g900"), "velvet")
            .when(F.col("USER_AGENT").contains("smart-tv"), "smart_tv")
            .when(F.col("USER_AGENT").contains("MARX-L"), "P30_lite")
            .when(
                F.col("USER_AGENT").contains("pixel"),
                F.regexp_extract(F.col("USER_AGENT"), "(pixel)(\\s)(\\d+)", 0),
            )
            .when(
                F.col("USER_AGENT").contains("redmi"),
                F.regexp_extract(F.col("USER_AGENT"), "(redmi)(\\s\\w+)*{2}", 0),
            )
            .when(
                F.col("USER_AGENT").contains(" mi "),
                F.regexp_extract(F.col("USER_AGENT"), "(mi)(\\s\\w+)*{2}", 0),
            ),
        )
        .withColumn(
            "OS_NAME",
            F.when(F.col("USER_AGENT").contains("iphone"), "ios")
            .when(F.col("USER_AGENT").contains("ipad"), "ipados")
            .when(F.col("USER_AGENT").contains("android"), "android")
            .when(F.col("USER_AGENT").contains("windows"), "windows")
            .when(F.col("USER_AGENT").contains("macintosh"), "macos")
            .when(
                (F.col("USER_AGENT").contains("ubuntu"))
                | (F.col("USER_AGENT").contains("CrOS"))
                | (F.col("USER_AGENT").contains("x86_64")),
                "linux",
            ),
        )
        .withColumn(
            "OS_VERSION",
            F.when(
                F.col("USER_AGENT").contains("android"),
                F.regexp_extract(F.col("USER_AGENT"), "android (\\d+)", 1),
            )
            .when(
                F.col("USER_AGENT").contains("iphone"),
                F.regexp_extract(F.col("USER_AGENT"), "(\\d+)(_)(\\d)", 0),
            )
            .when(
                F.col("USER_AGENT").contains("windows"),
                F.regexp_extract(F.col("USER_AGENT"), "nt (\\d+)", 1),
            ),
        )
        .withColumn(
            "DEVICE_CATEGORY",
            F.when(F.col("OS_NAME").isin(desktop_os), "desktop")
            .when(
                (F.col("OS_NAME") == "ios")
                | (
                    (F.col("OS_NAME") == "android")
                    & (
                        (~F.col("DEVICE_MARKETING_NAME").isin(tablet_names))
                        | (F.col("DEVICE_MARKETING_NAME").isNull())
                        | (F.col("DEVICE_BRAND_NAME").isNull())
                    )
                ),
                "mobile",
            )
            .when(F.col("DEVICE_MARKETING_NAME").isin(tablet_names), "tablet")
            .when(F.col("DEVICE_MARKETING_NAME") == "smart_tv", "TV"),
        )
        .withColumn(
            "BROWSER_NAME",
            F.when(F.col("USER_AGENT").contains("firefox"), "mozilla")
            .when(
                (F.col("USER_AGENT").contains("mac"))
                & (~F.col("USER_AGENT").contains("chrome"))
                & (~F.col("USER_AGENT").contains("firefox"))
                & (~F.col("USER_AGENT").contains("edg"))
                & (~F.col("USER_AGENT").contains("sznprohlizec"))
                & (~F.col("USER_AGENT").contains("opr"))
                & (~F.col("USER_AGENT").contains("crios")),
                "safari",
            )
            .when(F.col("USER_AGENT").contains("edg"), "edge")
            .when(F.col("USER_AGENT").contains("opr"), "opera")
            .when(F.col("USER_AGENT").contains("sznprohlizec"), "seznam")
            .when((F.col("USER_AGENT").isNull()) | (F.col("USER_AGENT") == ""), None)
            .otherwise("chrome"),
        )
        .withColumn(
            "DEVICE_MARKETING_NAME",
            F.regexp_replace("DEVICE_MARKETING_NAME", "build", ""),
        )
    )


df_calculate_device_feature = calculate_device_features(df_filtered_data)

# COMMAND ----------

# MAGIC %md Process new records

# COMMAND ----------


def extract_url_formats(df: DataFrame):
    # NORMALIZED URL
    return df_url_normalization(
        df.withColumn("URL_NORMALIZED", F.col("URL")), "URL_NORMALIZED"
    )


df_extract_url_formats = extract_url_formats(df_calculate_device_feature)

# COMMAND ----------


def order_into_sessions(df: DataFrame):
    df = df.repartition(F.col("DEVICE"))
    window_spec = Window.partitionBy("DEVICE").orderBy("TIMESTAMP")
    df = (
        df.withColumn("event_last", F.lag("TIMESTAMP", 1).over(window_spec))
        .withColumn(
            "event_difference",
            F.unix_timestamp("TIMESTAMP") - F.unix_timestamp("event_last"),
        )
        .withColumn("diff_in_minutes", F.col("event_difference") / 60)
        .withColumn("splits", (F.col("diff_in_minutes") >= 30).cast("integer"))
        .fillna(value=0, subset=["splits", "event_difference"])
        .withColumn("TIME_GROUPING", F.sum("splits").over(window_spec))
        .fillna(value=0, subset=["diff_in_minutes"])
    )

    df = df.repartition(F.col("DEVICE"), F.col("TIME_GROUPING"))
    window_spec2 = Window.partitionBy("DEVICE", "TIME_GROUPING").orderBy("TIMESTAMP")
    df = (
        df.withColumn("event_last", F.lag("TIMESTAMP", 1).over(window_spec2))
        .withColumn(
            "event_difference",
            F.unix_timestamp("TIMESTAMP") - F.unix_timestamp("event_last"),
        )
        .withColumn("diff_in_minutes", F.col("event_difference") / 60)
        .fillna(value=0, subset=["diff_in_minutes"])
        .withColumn("session_duration", F.sum("diff_in_minutes").over(window_spec2))
        .withColumn("REFERER", F.first("REFERER").over(window_spec2))
    )

    return df.withColumn(
        "zipped_events",
        F.array(
            "URL",
            "TIMESTAMP",
            "URL_NORMALIZED",
            "OWNER_ID",
            "OWNER_NAME",
            "SEARCH_ENGINE",
            "FLAG_ADVERTISER",
            "FLAG_PUBLISHER",
        ),
    )


df_order_into_session = order_into_sessions(df_extract_url_formats)

# COMMAND ----------


def create_session_id(df: DataFrame):
    """
    Return DataFrame with unique session identifier.
    Use xxhash64 hash function to ensure unique identifier for each session.
    Ensure one-to-one relationship with session and its dimension tables - e.g. device.
    """
    return df.withColumn(
        "SESSION_ID",
        sdm_hash(
            "DEVICE",
            "TIME_GROUPING",
            "DEVICE_CATEGORY",
            "DEVICE_BRAND_NAME",
            "DEVICE_MARKETING_NAME",
            "OS_NAME",
            "OS_VERSION",
            "BROWSER_NAME",
        ),
    )


df_create_session_id = create_session_id(df_order_into_session)

# COMMAND ----------


def group_into_sessions(df: DataFrame):
    grouped_sessions = df.groupBy(
        "DEVICE",
        "SESSION_ID",
        "DEVICE_CATEGORY",
        "DEVICE_BRAND_NAME",
        "DEVICE_MARKETING_NAME",
        "OS_NAME",
        "OS_VERSION",
        "BROWSER_NAME",
    ).agg(
        F.max("session_duration").alias("TOTAL_TIME_IN_SESSION"),
        F.min("TIMESTAMP").alias("SESSION_START_TIME"),
        F.max("TIMESTAMP").alias("SESSION_END_TIME"),
        F.min("TIMESTAMP").alias("TIMESTAMP"),
        F.count(F.lit(1)).alias("NUMBER_OF_EVENTS"),
        F.first("IP_ADDRESS").alias("IPv4"),
        F.first("DATE").alias("DATE"),
        F.collect_list("zipped_events").alias("PAGES_TIMES"),
        F.countDistinct("URL").alias("UNIQUE_PAGE_VIEWS"),
        F.first("REFERER").alias("REFERER"),
    )

    grouped_sessions = grouped_sessions.select(
        "*", F.col("NUMBER_OF_EVENTS").alias("NUMBER_OF_PAGE_VIEWS")
    )

    grouped_sessions = df_url_normalization(
        grouped_sessions.withColumn("REFERER_NORMALIZED", F.col("REFERER")),
        "REFERER_NORMALIZED",
    )
    grouped_sessions = grouped_sessions.withColumnRenamed("DEVICE", "USER_ID")

    return grouped_sessions


df_group_into_sessions = group_into_sessions(df_create_session_id)

# COMMAND ----------


def add_empty_cols_and_zipped_device(df: DataFrame):
    return (
        df.withColumnRenamed("REFERER_NORMALIZED", "REFERRAL_PATH_NORMALIZED")
        .withColumnRenamed("REFERER", "REFERRAL_PATH_FULL")
        .withColumn("BROWSER_LANGUAGE", F.lit(None).cast(T.StringType()))
        .withColumn("VIEW_PORT_SIZE", F.lit(None).cast(T.StringType()))
        .withColumn("SCREEN_SIZE", F.lit(None).cast(T.StringType()))
        .withColumn("SESSION_NUMBER", F.lit(None).cast(T.StringType()))  # To be DONE
        .withColumn("IS_BOUNCED", F.lit(None).cast(T.StringType()))  # To be DONE
        .withColumn("TRAFFIC_SOURCE_SOURCE", F.lit(None).cast(T.StringType()))
        .withColumn("TRAFFIC_SOURCE_MEDIUM", F.lit(None).cast(T.StringType()))
        .withColumn("TRAFFIC_SOURCE_KEYWORDS", F.lit(None).cast(T.StringType()))
        .withColumn("TRAFFIC_SOURCE_AD_CONTENT", F.lit(None).cast(T.StringType()))
        .withColumn("CHANNELING_GROUP", F.lit(None).cast(T.StringType()))
        .withColumn("LOCATION_ID", F.lit(None).cast(T.StringType()))
        .withColumn("IPv6", F.lit(None).cast(T.StringType()))
        .withColumn("OS_LANGUAGE", F.lit(None).cast(T.StringType()))
        .withColumn("PAGEVIEW_TYPE", F.lit(None).cast(T.StringType()))
        .withColumn("DURATION", F.lit(None).cast(T.DoubleType()))
        .withColumn(
            "ZIPPED_DEVICE_DETAILS",
            F.array(
                "DEVICE_CATEGORY",
                "DEVICE_BRAND_NAME",
                "DEVICE_MARKETING_NAME",
                "OS_NAME",
                "OS_VERSION",
                "BROWSER_NAME",
            ),
        )
    )


df_add_empty_cols_and_zipped_device = add_empty_cols_and_zipped_device(
    df_group_into_sessions
)

# COMMAND ----------


def create_foreign_dimension_keys(df: DataFrame):
    """Return DataFrame with hashed foreign keys for dimension tables"""
    return (
        df.withColumn("BROWSER_ID", sdm_hash("BROWSER_NAME"))
        .withColumn(
            "DEVICE_ID",
            sdm_hash("DEVICE_CATEGORY", "DEVICE_BRAND_NAME", "DEVICE_MARKETING_NAME"),
        )
        .withColumn("OS_ID", sdm_hash("OS_NAME", "OS_VERSION"))
    )


df_create_foreign_dimension_keys = create_foreign_dimension_keys(
    df_add_empty_cols_and_zipped_device
)

# COMMAND ----------

# MAGIC %md #### Save preprocessed table

# COMMAND ----------


def save_preprocessed_table(df: DataFrame):
    return df.select(
        "USER_ID",
        "SESSION_ID",
        "BROWSER_ID",
        "BROWSER_LANGUAGE",
        "VIEW_PORT_SIZE",
        "SCREEN_SIZE",
        "SESSION_NUMBER",
        "SESSION_START_TIME",
        "SESSION_END_TIME",
        "IS_BOUNCED",
        "NUMBER_OF_EVENTS",
        "NUMBER_OF_PAGE_VIEWS",
        "UNIQUE_PAGE_VIEWS",
        "TOTAL_TIME_IN_SESSION",
        "REFERRAL_PATH_FULL",
        "REFERRAL_PATH_NORMALIZED",
        "TRAFFIC_SOURCE_SOURCE",
        "TRAFFIC_SOURCE_MEDIUM",
        "TRAFFIC_SOURCE_KEYWORDS",
        "TRAFFIC_SOURCE_AD_CONTENT",
        "CHANNELING_GROUP",
        "LOCATION_ID",
        "IPv6",
        "IPv4",
        "DEVICE_ID",
        "ZIPPED_DEVICE_DETAILS",
        "OS_ID",
        "OS_LANGUAGE",
        "DATE",
        "TIMESTAMP",
        "PAGES_TIMES",
        "DURATION",
        "PAGEVIEW_TYPE",
    )


df_save_preprocessed_table = save_preprocessed_table(df_create_foreign_dimension_keys)

schema_sdm_preprocessed, info_sdm_preprocessed = get_schema_sdm_preprocessed()

write_dataframe_to_table(
    df_save_preprocessed_table,
    config.paths.sdm_preprocessed,
    schema_sdm_preprocessed,
    "overwrite",
    root_logger,
    info_sdm_preprocessed["partition_by"],
    info_sdm_preprocessed["table_properties"],
)
