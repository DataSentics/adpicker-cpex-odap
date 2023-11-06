from pyspark.sql.session import SparkSession
from pyspark.dbutils import DBUtils


def getMySqlOptions(dbutils: DBUtils):
    dbConfig = {
        "host": dbutils.secrets.get(
            scope="unit-kv", key="personas-db-sql-server-host-name"
        ),
        "user": dbutils.secrets.get(scope="unit-kv", key="personas-db-admin-username"),
        "password": dbutils.secrets.get(
            scope="unit-kv", key="personas-db-admin-password"
        ),
        "database": dbutils.secrets.get(
            scope="unit-kv", key="personas-db-database-name"
        ),
        "port": dbutils.secrets.get(scope="unit-kv", key="personas-db-server-port"),
        "timezone": "UTC",
    }

    url = "jdbc:mysql://{}:{}/{}?serverTimezone={}".format(
        dbConfig["host"],
        dbConfig["port"],
        dbConfig["database"],
        dbConfig["timezone"],
    )

    return dict(
        url=url,
        driver="com.mysql.cj.jdbc.Driver",
        user=dbConfig["user"],
        password=dbConfig["password"],
    )


def load_mysql_table(table_name, spark: SparkSession, dbutils: DBUtils):
    """
    Loads table from MySQL db - personas.
    """
    return (
        spark.read.format("jdbc")
        .options(**getMySqlOptions(dbutils))
        .option("dbtable", table_name)
        .load()
    )


def overwrite_mysql_table_by_df(df, table_name, spark: SparkSession, dbutils: DBUtils):
    """
    Overwrites MySQL table without dropping it (maintains the schema)
    """
    try:
        (
            df.write.format("jdbc")
            .mode("overwrite")
            .options(**getMySqlOptions(dbutils))
            .option("dbtable", table_name)
            .option("truncate", "true")
            .save()
        )
        return True
    except BaseException as e:
        print(e)
        return False


def append_df_to_mysql_table(df, table_name, spark: SparkSession, dbutils: DBUtils):
    try:
        (
            df.write.format("jdbc")
            .mode("append")
            .options(**getMySqlOptions(dbutils))
            .option("dbtable", table_name)
            .save()
        )
        return True
    except BaseException as e:
        print(e)
        return False
