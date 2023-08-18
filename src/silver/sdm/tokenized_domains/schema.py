from pyspark.sql import types as T


def get_schema():
    schema = T.StructType(
        [
            T.StructField("USER_ID", T.StringType(), True),
            T.StructField("TOKENS", T.ArrayType(T.StringType()), True),
            T.StructField("URL_NORMALIZED", T.StringType(), True),
            T.StructField("DATE", T.DateType(), True),
            T.StructField("TOKEN", T.StringType(), True),
        ])

    info ={
    "primary_key":[],
    "partition_by":['DATE'], # INSERT PARTITION KEY(s) HERE (OPTIONAL)
    "table_properties": {
        'delta.autoOptimize.optimizeWrite': 'true'
    }
        }
    return schema, info