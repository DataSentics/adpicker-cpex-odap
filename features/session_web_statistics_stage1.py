# Databricks notebook source
# MAGIC %md
# MAGIC # Session features - web statistics
# MAGIC This notebook creates the following features from `sdm_session` table, all in chosen last *n*-day window:
# MAGIC - web_analytics_time_on_site_average
# MAGIC - web_analytics_web_active
# MAGIC - web_analytics_web_security_affinity

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #### Imports & config

# COMMAND ----------

import pyspark.sql.functions as F
import re

from pyspark.sql.window import Window

from src.utils.helper_functions_defined_by_user.yaml_functions import (
    get_value_from_yaml,
)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #### Load table & fetch config values

# COMMAND ----------

df_sdm_session =  spark.read.format("delta").load(
    get_value_from_yaml("paths", "sdm_table_paths", "sdm_session")
)

time_window_str = get_value_from_yaml("featurestorebundle", "time_windows")[0]
time_window_int = int(re.search(r'\d+', time_window_str).group())

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Filter table

# COMMAND ----------

df_session_filtered = df_sdm_session.filter(F.col("session_date") >= F.current_date() - time_window_int)

# COMMAND ----------

def calculate_web_statistics(df):
    df_temp_features = df.withColumn(
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
df_web_statistics.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Metadata

# COMMAND ----------

metadata = {
    "table": "user",
    "category": "digital_device",
    "features": {
        f"web_analytics_time_on_site_average_{time_window_str}": {
            "description": f"Average duration of the web session in last {time_window_str} expressed in seconds.",
            "fillna_with": 0,
        },
        f"web_analytics_web_active_{time_window_str}": {
            "description": f"Indicator whether the client was active on web in last {time_window_str}.",
            "fillna_with": 0,
        },
        f"web_analytics_web_security_affinity_{time_window_str}": {
            "description": f"Score of how much the client cares about web security in last {time_window_str}.",
            "fillna_with": 0,
        },
    },
}
