


from typing import Optional, List
from logging import Logger, getLogger
from pyspark.sql import SparkSession

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
    spark = SparkSession.builder.appName("MyApp").getOrCreate()
    
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
        raise ValueError("!! Data frame schema doesn't match schema of table you try to write in !!")


def delta_table_exists(table_path):
    """
    This function checks if a delta table exists

    Parameters:
    table_path - the full path to the delta table(including its name)
    """

    try:
        DeltaTable.forPath(spark, table_path)
        return True
    except:
        return False
    
