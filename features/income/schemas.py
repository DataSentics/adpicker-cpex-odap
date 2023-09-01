from pyspark.sql import types as T


def get_income_interest_scores():
    # Define the schema
    schema = T.StructType( [
            T.StructField('user_id', T.StringType(),True),
            T.StructField('timestamp', T.TimestampType(),True),
            T.StructField('final_interest_score_low', T.DoubleType(),True),
            T.StructField('final_interest_score_mid', T.DoubleType(),True),
            T.StructField('final_interest_score_high', T.DoubleType(),True),
        ])
    
    # Define additional information
    info ={
        "primary_key": [''],
        "partition_by": [''], # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {
            'delta.autoOptimize.optimizeWrite': 'true'
            }
    }
    return schema, info

def get_income_other_scores():
    # Define the schema
    schema = T.StructType( [
            T.StructField('user_id', T.StringType(),True),
            T.StructField('final_other_score_low', T.DoubleType(),True),
            T.StructField('final_other_score_mid', T.DoubleType(),True),
            T.StructField('final_other_score_high', T.DoubleType(),True),
            T.StructField('timestamp', T.TimestampType(),True),
        ])
    
    # Define additional information
    info ={
        "primary_key": [''],
        "partition_by": [''], # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {
            'delta.autoOptimize.optimizeWrite': 'true'
            }
    }
    return schema, info


def get_income_url_scores():
    # Define the schema
    schema = T.StructType( [
            T.StructField('user_id', T.StringType(),True),
            T.StructField('timestamp', T.TimestampType(),True),
             T.StructField('collected_urls', T.ArrayType(T.StringType(),True), True),
            T.StructField('final_other_score_low', T.DoubleType(),True),
            T.StructField('final_url_score_mid', T.DoubleType(),True),
            T.StructField('final_url_score_high', T.DoubleType(),True),
            
        ])
    
    # Define additional information
    info ={
        "primary_key": [''],
        "partition_by": [''], # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {
            'delta.autoOptimize.optimizeWrite': 'true'
            }
    }
    return schema, info

