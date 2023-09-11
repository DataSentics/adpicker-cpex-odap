from pyspark.sql import types as T

def get_education_interest_scores():
    # Define the schema
    schema = T.StructType( [
            T.StructField('user_id', T.StringType(),True),
            T.StructField('timestamp', T.TimestampType(),True),
            T.StructField('final_interest_score_zs', T.DoubleType(),True),
            T.StructField('final_interest_score_ss_no', T.DoubleType(),True),
            T.StructField('final_interest_score_ss_yes', T.DoubleType(),True),
            T.StructField('final_interest_score_vs', T.DoubleType(),True),
        ])
    
    # Define additional information
    info ={
        "primary_key": ['SESSION_ID'],
        "partition_by": ['DATE'], # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {
            'delta.autoOptimize.optimizeWrite': 'true'
            }
    }
    return schema, info


def get_education_other_scores():
       # Define the schema
    schema = T.StructType( [
            T.StructField('user_id', T.StringType(),True),
            T.StructField('final_other_score_zs', T.DoubleType(),True),
            T.StructField('final_other_score_ss_no', T.DoubleType(),True),
            T.StructField('final_other_score_ss_yes', T.DoubleType(),True),
            T.StructField('final_other_score_vs', T.DoubleType(),True),
            T.StructField('timestamp', T.TimestampType(),True)
        ])
    
    # Define additional information
    info ={
        "primary_key": ['SESSION_ID'],
        "partition_by": ['DATE'], # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {
            'delta.autoOptimize.optimizeWrite': 'true'
            }
    }
    return schema, info



def get_education_url_scores():
       # Define the schema
    schema = T.StructType( [
            T.StructField('user_id', T.StringType(),True),
            T.StructField('timestamp', T.TimestampType(),True),
            T.StructField('collected_urls', T.ArrayType(T.StringType(),True), True),
            T.StructField('final_url_score_zs', T.DoubleType(),True),
            T.StructField('final_url_score_ss_no', T.DoubleType(),True),
            T.StructField('final_url_score_ss_yes', T.DoubleType(),True),
            T.StructField('final_url_score_vs', T.DoubleType(),True),
            
        ])
    
    # Define additional information
    info ={
        "primary_key": ['SESSION_ID'],
        "partition_by": ['DATE'], # INSERT PARTITION KEY(s) HERE (OPTIONAL)
        "table_properties": {
            'delta.autoOptimize.optimizeWrite': 'true'
            }
    }
    return schema, info