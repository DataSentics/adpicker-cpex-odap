from pyspark.sql import types as T


def get_lookalike_schema():
    schema = T.StructType(
        [
            T.StructField("Id", T.IntegerType(), True),
            T.StructField("Model", T.StringType(), True),
            T.StructField("account", T.StringType(), True),
            T.StructField("client_name", T.StringType(), True),
            T.StructField("TP_DMP_id", T.StringType(), True),
            T.StructField("TP_DMP_type", T.StringType(), True),
            T.StructField("next_retraining", T.StringType(), True),
            T.StructField("segment_id", T.StringType(), True),
            T.StructField("days_before_tp", T.IntegerType(), True),
        ]
    )

    info = {
        "primary_key": ["Id"],
        "partition_by": [],  # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {"delta.autoOptimize.optimizeWrite": "true"},
    }

    return schema, info
