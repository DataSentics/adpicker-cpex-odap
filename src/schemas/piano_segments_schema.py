from pyspark.sql import types as T


def get_piano_segments_schema():
    schema = T.StructType(
        [
            T.StructField("user_id", T.StringType()),
            T.StructField("segment_id", T.StringType()),
        ]
    )

    info = {
        "primary_key": ["user_id", "segment_id"],
        "partition_by": ["segment_id"],  # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {"delta.autoOptimize.optimizeWrite": "true"},
    }

    return schema, info