from pyspark.sql import types as T

# schema - cleansed
def get_schema_user_traits():
    schema = T.StructType([
            T.StructField('USER_ID', T.StringType(),True),
            T.StructField('TRAIT', T.StringType(),True),
            T.StructField('FIRST_DATE', T.DateType(),True),
            T.StructField('RUN_DATE', T.DateType(),False),
            T.StructField('N_DAYS', T.IntegerType(),True),
            T.StructField('END_DATE', T.DateType(),False),
        ])

    info ={
        "primary_key": ['USER_ID', 'TRAIT'],
        "partition_by": [], # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {
            'delta.autoOptimize.optimizeWrite': 'true'
            }
    }

    return schema, info

def get_schema_user_segments():
    schema = T.StructType([
            T.StructField('USER_ID', T.StringType(),True),
            T.StructField('SEGMENT', T.StringType(),True),
            T.StructField('FIRST_DATE', T.DateType(),True),
            T.StructField('RUN_DATE', T.DateType(),False),
            T.StructField('N_DAYS', T.IntegerType(),True),
            T.StructField('END_DATE', T.DateType(),False),
        ])

    info ={
        "primary_key": ['USER_ID', 'SEGMENT'],
        "partition_by": [], # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {
            'delta.autoOptimize.optimizeWrite': 'true'
            }
    }

    return schema, info

