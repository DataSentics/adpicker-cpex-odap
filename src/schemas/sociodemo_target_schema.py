from pyspark.sql import types as T


def get_sociodemo_target_schema():
    schema = T.StructType(
        [
            T.StructField('AGE', T.StringType(),True),
            T.StructField('GENDER', T.StringType(),True),
            T.StructField('USER_ID', T.StringType(),True),
            T.StructField('PUBLISHER', T.StringType(),True),
            T.StructField('TIMESTAMP', T.TimestampType(),True),
            T.StructField('DATE', T.DateType(),True)
        ])

    info ={
    "primary_key":['USER_ID', 'TIMESTAMP'], # ?CONTAINER_ID?
    "partition_by":['DATE'], # INSERT PARTITION KEY(s) HERE (OPTIONAL)
    "table_properties": {
        'delta.autoOptimize.optimizeWrite': 'true'
        }
        }
    return schema, info
