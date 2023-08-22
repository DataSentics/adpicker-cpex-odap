from typing import Optional, List
from logging import Logger, getLogger
from pyspark.sql import SparkSession
from delta.tables import DeltaTable

spark = SparkSession.builder.appName("MyApp").getOrCreate()

# Create a Spark session
# will put the data from dataframe in delta table in th specified mode, also specifying metadata (partition and table properties)
def write_dataframe_to_table(dataframe_source, table_destination, table_schema, writing_mode, partition: Optional[List[str]] = None, table_properties: Optional[str] = None):

    """
    This function takes 6 parameters, first 4 mandatory last two optional, and writes the data from a dataframe into a table.
    Data will be written in the table if schema of dataframe is compatible with schema of table otherwise an error will be thrown 

    Parameters:
    dataframe_source - is the dataframe from which we want to write data into the table 
    table_destination - is the path to the table we want to write data in 
    writing_mode - the way we want to write data into the table (default/overwrite/append)
    partition - used to specify how you want the data to be partitioned within the storage system (optional)
    table_properties - set the metadata of the table (optional)

    """

    root_logger = getLogger()
    
    
    if dataframe_source.schema == table_schema: # the schema of dataframe from where we want to write data 

        if partition is None: # some tables don't have a parameter for partition
            dataframe_source.write.format("delta").mode(f"{writing_mode}").save(f"{table_destination}")
        else:
            dataframe_source.write.format("delta").mode(f"{writing_mode}").partitionBy(*partition).save(f"{table_destination}")

        if table_properties is not None: # some tables don't have parameter for table properties
            table_properties_str = ', '.join([f"'{key}' = '{value}'" for key, value in table_properties.items()])
            spark.sql( f"ALTER TABLE delta.`{table_destination}` SET TBLPROPERTIES ({table_properties_str})")
           

    else:

        root_logger.error("Schema mismatch")
        compare_and_check_schemas(table_schema, dataframe_source.schema)
        raise ValueError("!! Data frame schema doesn't match schema of table you try to write in !!")
        


def compare_and_check_schemas(schema1, schema2):
    """
    This function will print a message with what is the difference between the table schema and dataframe schema 

    Parameters:
    schema1 - the schema of table in which we want to write data 
    schema2 - the schema of dataframe from which we want to write the data in table 

    """
    fields1 = {field.name: field for field in schema1.fields}
    fields2 = {field.name: field for field in schema2.fields}
    mismatch_fields = []
    
    for field_name in fields1.keys() | fields2.keys():
        field1 = fields1.get(field_name)
        field2 = fields2.get(field_name)
        
        if field1 is None:
            print(f"FIELD: '{field_name}' is present in DATAFRAME schema but not in TABLE schema.")
        elif field2 is None:
            print(f"FIELD: '{field_name}' is present in TABLE schema but not in DATAFRAME.")
        else:
            if field1.dataType != field2.dataType:
                print(f"!! DataType mismatch !! TABLE schema: '{field_name}': {field1.dataType} | DATAFRAME schema: {field2.dataType} ")
            if field1.nullable != field2.nullable:
                print(f"!! Nullability mismatch !! TABLE schema: '{field_name}': {field1.nullable} | DATAFRAME schema: {field2.nullable}")


def delta_table_exists(table_path):
    """
    This function checks if a delta table exists

    Parameters:
    table_path - the full path to the delta table(including its name)
    """
    try:
        delta_table = DeltaTable.forPath(spark, table_path)
        return True
    except Exception as e:
        return False
    

