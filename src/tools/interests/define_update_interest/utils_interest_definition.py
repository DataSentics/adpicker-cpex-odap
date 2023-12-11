import pandas as pd
from logging import Logger

from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F


from src.utils.helper_functions_defined_by_user._functions_udf import udf_values_count_str


# -------------------------------- token frequency computation --------------------------------

def get_most_frequent_per_column(df: DataFrame, top_n: int, columns: list = None, logger: Logger = None):
    """
    Returns a dataframe with most frequent values in each column 
    (number of columns & their names remain unchanged)
    
    The values are strings "<value>: <frequency>"
    """
    if columns is None: 
        columns = df.columns
        
    df_result = pd.DataFrame(index=pd.RangeIndex(stop=top_n))
    # display most frequent tokens for each columns separately
    for col in columns:
        # find most frequent tokens in current column
        df_cnt_curr = (df
                       .withColumn("token", F.explode(col))
                       .groupBy("token")
                       .agg(F.count(col).alias("count"))
                       .orderBy(F.col("count").desc())
                       .limit(top_n)
                       .toPandas()
                      )
        # append result
        if df_cnt_curr.empty: 
            if logger is not None:
                logger.warning(f"Could not aggregate most frequent values for '{col}' - there are no tokens in the column.")
        else:
            df_result[col] = df_cnt_curr.apply(lambda x: f"{x.loc['token']}: {x.loc['count']}", axis=1)
    
    return df_result

# -------------------------------- JLH score computation --------------------------------

# TODO: it would be nice to refactor this function as it is extremely unreadable

def jlh(df,
        data_col: str,
        doc_id_col: str,
        power_coef: float = 1,
        minDF: int = 3,
        ignore_duplicates: bool = True,
        label: str = 'LABEL'
       ):
    """
    Computes JLH score for each token in dataframe data column. 
    
    -----------------------------------------------------------------------------------------------
    JLH score = (foreground / background) * (foreground - background)
    Foreground: out of all (distinct) queries within a category, what proportion contains the token
      = P(token | category)
    Background: out of all (dictinct) queries overall, what proportion contains the token
      = P(token)
      
    description source (and maybe more readable implementation): 
    https://dbc-8c8038b8-0e28.cloud.databricks.com/#notebook/848185/command/848222
    
    :param data_col: column with the collections that should be scored
    :param doc_id_col: grouping ID 
    :param power_coef: power coefficient of the lift (= foreground / background) - the higher, the more important the lift is
    :param minDF: minimal number of occurences in the data_col for a data point to be included in the result
    :param ignore_duplicates: ignore duplicates in the data columns
    :param label: foreground flag column (0 / 1)
    """

    if minDF < 0:
        raise ValueError(f"Invalid 'minDF' value. Expected >= 0, got '{minDF}'")

    if power_coef <= 0:
        raise ValueError(f"Invalid 'power_coef' value. Expected >= 0, got '{power_coef}'")

    df_dataset = (df
                  .withColumn('label',F.col(label))
                  .select(doc_id_col, data_col, "label")
                  .withColumn("token", F.explode(F.col(data_col)))
                  )
    
    df_dataset = df_dataset.filter(df_dataset.token.rlike('\D+'))

    if ignore_duplicates:
        count_all = df_dataset.groupBy().agg(F.countDistinct(doc_id_col)).collect()[0][0]
        count_target = df_dataset.filter(F.col("label") == 1).groupBy().agg(F.countDistinct(doc_id_col)).collect()[0][0]
    else:
        count_all = df_dataset.count()
        count_target = df_dataset.filter(F.col("label") == 1).count()

    df_background_set = (df_dataset
                         .groupBy("token")
                         .agg(F.count("*").alias("token_count"),
                              F.countDistinct(doc_id_col).alias("token_users")
                              )
                         .filter(F.col("token_users") > minDF)
                         .withColumn("background_pc",
                                     (F.col("token_users" if ignore_duplicates else "token_count") / count_all) * 100
                                     )
                         )

    df_foreground_set = (df_dataset
                         .filter(F.col("label") == 1)
                         .groupby("token")
                         .agg(F.countDistinct(doc_id_col).alias("token_users_target"),
                              F.count("*").alias("token_count_target")
                              )
                         .withColumn("foreground_pc",
                                     (F.col("token_users_target" if ignore_duplicates else "token_count_target") / count_target) * 100
                                     )
                         )

    df_jlh_set = (df_background_set
                  .join(df_foreground_set, "token", "inner")
                  .select("token", "token_users_target", "token_users", "foreground_pc", "background_pc")
                  )

    df_jlh = (df_jlh_set
              .withColumn("power_lift", pow(F.col("foreground_pc") / F.col("background_pc"), power_coef))
              .withColumn("diff", F.col("foreground_pc") - F.col("background_pc"))
              .withColumn("JLH_score", F.col("power_lift") * F.col("diff"))
              )

    return df_jlh

# -------------------------------- URL hits computation --------------------------------

def url_hits_for_group(df: DataFrame, data_col: str, group_by_col: str):
    """
    Aggregates number of hits per URL and breaks the hits count down into hits per each keyword.
    
    Expects the `data_col` to be a single value column (exploded array-type column)
    """
    # new & temporary columns definition
    col_collected = f"{data_col}_COLLECTED_"
    hits_total_col = "N_HITS"
    hits_per_token_col = "HITS_PER_VALUE"
    # collect all values per the group -> count them & append values count 
    df_cnt = (df
              .groupBy(group_by_col)
              .agg(F.collect_list(data_col).alias(col_collected))
              .withColumn(hits_total_col, F.size(col_collected))
              .withColumn(hits_per_token_col, udf_values_count_str(col_collected))
              .drop(col_collected)
              .orderBy(F.desc(hits_total_col))
             )
    
    return df_cnt


