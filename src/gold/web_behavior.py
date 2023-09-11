# Databricks notebook source
# MAGIC %md 
# MAGIC ### Imports & widgets

# COMMAND ----------

from functools import reduce
from operator import add
from pyspark.sql.window import Window

import pyspark.sql.functions as F

# COMMAND ----------

dbutils.widgets.text("time_window", "7", "Time window")
time_window_int = int(getArgument("time_window"))
time_window_str = f"{getArgument('time_window')}d"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Load tables

# COMMAND ----------

def load_session_filtered():
    path = "/mnt/aam-cpex-dev/solutions/testing/sdm_session.delta"
    df = (
        spark.read.format("delta")
        .load(path)
        .filter(F.col("session_date") >= (F.current_date() - F.lit(time_window_int)))
    )
    return df


df_session_filtered = load_session_filtered()

# COMMAND ----------

def load_pageview_filtered():
    path = "/mnt/aam-cpex-dev/solutions/testing/sdm_pageview.delta"
    df = (
        spark.read.format("delta")
        .load(path)
        .filter(F.col("page_screen_view_date") >= (F.current_date() - F.lit(time_window_int)))
    )
    return df


df_pageview_filtered = load_pageview_filtered()

# COMMAND ----------

def web_with_visit_period(df_session_filterd):
    return df_session_filtered.withColumn(
        "visit_hour", F.hour(F.col("session_start_datetime"))
    ).withColumn(
        "visit_period",
        F.when(F.col("visit_hour").between(5, 8), "Early morning")
        .when(F.col("visit_hour").between(9, 11), "Late morning")
        .when(F.col("visit_hour").between(12, 15), "Early afternoon")
        .when(F.col("visit_hour").between(16, 17), "Late afternoon")
        .when(F.col("visit_hour").between(18, 20), "Early evening")
        .when(F.col("visit_hour").between(21, 23), "Late evening")
        .otherwise("Night"),
    )


df_session_with_visit_period = web_with_visit_period(df_session_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Most common visit period

# COMMAND ----------

def calculate_visit_time_most_common(df_session_with_visit_period):
    df = df_session_with_visit_period.groupby("user_id").agg(
        F.mode("visit_period").alias(
            f"web_analytics_visit_time_most_common_{time_window_str}"
        )
    )
    return df


df_visit_time = calculate_visit_time_most_common(df_session_with_visit_period)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Percentage of device visits

# COMMAND ----------

def calculate_device_percentage(df_session_filtered):
    device_list = ["mobile", "desktop", "smart_tv", "tablet"]
    df_grouped = df_session_filtered.groupby("user_id").agg(
        *[
            (F.count(F.when(F.col("device_category") == device, 1))).alias(
                f"count_{device}_{time_window_str}"
            )
            for device in device_list
        ]
    )
    df_summed = df_grouped.withColumn(
        "total",
        reduce(
            add, [F.col(f"count_{device}_{time_window_str}") for device in device_list]
        ),
    )
    df_percentages = df_summed
    for device in device_list:
        df_percentages = df_percentages.withColumn(
            f"{device}_avg_users_{time_window_str}",
            F.round(F.col(f"count_{device}_{time_window_str}") / F.col("total"), 1),
        )
    cols_to_drop = ["total", *[f"count_{device}_{time_window_str}" for device in device_list]]
    df_final = df_percentages.drop(*cols_to_drop)
    return df_final


df_device_percentage = calculate_device_percentage(df_session_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Pageview total count

# COMMAND ----------

def calculate_total_pageview_count(df_pageview_filtered):
    df_pageview_count = df_pageview_filtered.groupby("user_id").agg(
        F.count(F.lit(1)).alias(f"web_analytics_pageviews_sum_{time_window_str}")
    )
    return df_pageview_count


df_pageview_counts = calculate_total_pageview_count(df_pageview_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Web statistics
# MAGIC Average session duration, indicator of activity, security affinity.

# COMMAND ----------

def calculate_web_statistics(df_session_filtered):
    df_temp_features = df_session_filtered.withColumn(
        "session_duration",
        F.unix_timestamp(F.col("session_end_datetime"))
        - F.unix_timestamp("session_start_datetime"),
    )

    df_grouped = df_temp_features.groupby("user_id").agg(
        F.round(F.mean("session_duration")).alias(
            f"web_analytics_time_on_site_average_{time_window_str}"
        ),
        F.lit(1).alias(f"web_analytics_web_active_{time_window_str}"),
        F.lit(1).alias(f"distinct_cookies_{time_window_str}"),
        F.min("session_start_datetime").alias(f"min_session_start_{time_window_str}"),
        F.max("session_start_datetime").alias(f"max_session_start_{time_window_str}"),
    )

    df_web_statistics = df_grouped.withColumn(
        f"web_analytics_web_security_affinity_{time_window_str}",
        F.round(
            F.tanh(
                F.col(f"distinct_cookies_{time_window_str}")
                / (
                    F.datediff(
                        F.col(f"max_session_start_{time_window_str}"),
                        F.col(f"min_session_start_{time_window_str}"),
                    )
                    + 1
                )
            ),
            2,
        ),
    )

    cols_to_drop = [f"distinct_cookies_{time_window_str}", f"min_session_start_{time_window_str}", f"max_session_start_{time_window_str}"]
    return df_web_statistics.drop(*cols_to_drop)


df_web_statistics = calculate_web_statistics(df_session_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Device types 

# COMMAND ----------

def calculate_device_type_statistics(df_session_filtered):
    window_spec = Window.partitionBy("user_id").orderBy(
        F.col("session_start_datetime").desc()
    )
    df_windowed = df_session_filtered.withColumn(
        f"device_category_last_used", F.max("device_category").over(window_spec)
    )

    df_grouped = df_windowed.groupby("user_id").agg(
        F.mode("device_category").alias(
            f"web_analytics_device_type_most_common_{time_window_str}"
        ),
        F.first("device_category_last_used").alias(
            f"web_analytics_device_type_last_used_{time_window_str}"
        ),
    )
    return df_grouped


df_device_types = calculate_device_type_statistics(df_session_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Browser choice

# COMMAND ----------

def calculate_browser_statistics(df_session_filtered):
    window_spec = Window.partitionBy("user_id").orderBy(
        F.col("session_start_datetime").desc()
    )
    df_windowed = df_session_filtered.withColumn(
        f"browser_last_used", F.max("browser_name").over(window_spec)
    )

    df_grouped = df_windowed.groupby("user_id").agg(
        F.mode("browser_name").alias(
            f"web_analytics_device_browser_most_common_{time_window_str}"
        ),
        F.first("browser_last_used").alias(
            f"web_analytics_device_browser_last_used_{time_window_str}"
        ),
    )
    return df_grouped


df_browsers = calculate_browser_statistics(df_session_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Search engine choice

# COMMAND ----------

def page_screen_with_search_engine(df_pageview_filtered):
    return df_pageview_filtered.withColumn(
        "search_engine",
        F.when(F.col("search_engine") == "", "")
        .when(F.col("search_engine").contains("seznam"), "seznam")
        .when(F.col("search_engine").contains("google"), "google")
        .when(F.col("search_engine").contains("centrum"), "centrum")
        .when(F.col("search_engine").contains("bing"), "bing")
        .when(F.col("search_engine").contains("yahoo"), "yahoo")
        .otherwise(None),
    )  # replacement of other categories by NULL 


df_pageview_with_search_engine = page_screen_with_search_engine(df_pageview_filtered)

# COMMAND ----------

def calculate_search_engine_statistics(df_pageview_with_search_engine):
    window_spec = Window.partitionBy("user_id").orderBy(
        F.col("page_screen_view_timestamp").desc()
    )
    df_windowed = df_pageview_with_search_engine.withColumn(
        f"search_engine_last_used", F.max("search_engine").over(window_spec)
    )

    df_grouped = df_windowed.groupby("user_id").agg(
        F.mode("search_engine").alias(
            f"web_analytics_page_search_engine_most_common_{time_window_str}"
        ),
        F.first("search_engine_last_used").alias(
            f"web_analytics_page_search_engine_last_used_{time_window_str}"
        ),
    )
    return df_grouped


df_search_engine = calculate_search_engine_statistics(df_pageview_with_search_engine)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### OS choice

# COMMAND ----------

def calculate_os_statistics(df_session_filtered):
    window_spec = Window.partitionBy("user_id").orderBy(
        F.col("session_start_datetime").desc()
    )
    df_windowed = df_session_filtered.withColumn(
        f"os_last_used", F.max("os_name").over(window_spec)
    )

    df_grouped = df_windowed.groupby("user_id").agg(
        F.mode("os_name").alias(
            f"web_analytics_device_os_most_common_{time_window_str}"
        ),
        F.first("os_last_used").alias(
            f"web_analytics_device_os_last_used_{time_window_str}"
        ),
    )
    return df_grouped

df_os = calculate_os_statistics(df_session_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Distinct device categories

# COMMAND ----------

def calculate_distinct_device_categories(df_session_filtered):
    df_grouped = df_session_filtered.groupby("user_id").agg(
        F.count_distinct("device_category").alias(
            f"web_analytics_num_distinct_device_categories_{time_window_str}"
        )
    )
    return df_grouped

df_distinct_device_categories = calculate_distinct_device_categories(
    df_session_filtered
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Distinct devices

# COMMAND ----------

def calculate_distinct_devices(df_session_filtered):
    distinct_columns = [
        "os_name",
        "device_category",
        "device_brand_name",
    ]
    df_grouped = (
        df_session_filtered.na.fill(
            "unknown", subset=["os_name", "device_category", "device_brand_name"]
        )
        .groupby("user_id")
        .agg(
            F.count_distinct(*distinct_columns).alias(
                f"web_analytics_channel_device_count_distinct_{time_window_str}"
            )
        )
    )
    return df_grouped


df_distinct_devices = calculate_distinct_devices(df_session_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Total distinct web page visits
# MAGIC Only web pages - URLs with "blog" are excluded.

# COMMAND ----------

def calculate_distinct_web_visits(df_pageview_filtered):
    df_grouped = (
        df_pageview_filtered.filter(~F.col("full_url").contains("blog"))
        .groupby("user_id")
        .agg(F.count_distinct("page_screen_view_timestamp").alias(f"web_analytics_total_visits_{time_window_str}"))
    )
    return df_grouped


df_distinct_web_visits = calculate_distinct_web_visits(df_pageview_filtered)
df_distinct_web_visits.display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Days since last blog visit

# COMMAND ----------

def calculate_days_since_blog_visit(df_pageview_filtered):
    df_days_column = df_pageview_filtered.withColumn(
        "days_since_blog_view",
        F.when(
            F.col("full_url").contains("blog"),
            F.datediff(F.current_date(), F.col("page_screen_view_date")),
        ).otherwise(None),
    )

    df_group = df_days_column.groupby("user_id").agg(
        F.min("days_since_blog_view").alias(
            f"web_analytics_blog_days_since_last_visit_{time_window_str}"
        )
    )
    return df_group

df_days_since_last_blog_visit = calculate_days_since_blog_visit(df_pageview_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Publishers list

# COMMAND ----------

def collect_publisher_list(df_pageview_filtered):
    df_grouped = df_pageview_filtered.groupby("user_id").agg(
        F.collect_set("owner_name").alias("owner_name_set")
    )
    df_result = df_grouped.withColumn(
        f"owner_names_{time_window_str}", F.concat_ws(",", F.col("owner_name_set"))
    )
    return df_result


df_publishers = collect_publisher_list(df_pageview_filtered)
df_publishers.display()
