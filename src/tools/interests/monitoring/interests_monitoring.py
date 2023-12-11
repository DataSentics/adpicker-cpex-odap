# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Interest monitoring
# MAGIC 
# MAGIC This notebook performs monitoring of all interests.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC There are 3 main parts of the monitoring:
# MAGIC * **Tokenized domains monitoring** - analysis of user's tokens which appear in interest definitions
# MAGIC * **Cross-interest monitoring** - uniqueness & conflicting keywords across interest definitions 
# MAGIC * **Data monitoring** - analysis which uses purely the pageview tokens (no interest definitions) 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ---

# COMMAND ----------

# MAGIC %run ./../../../app/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Imports

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Global imports:

# COMMAND ----------

# global imports
import daipe as dp

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from logging import Logger


# project-level imports
# TODO this should not be needed
from adpickercpex.lib.FeatureStoreTimestampGetter import FeatureStoreTimestampGetter
from adpickercpex.lib.display_result import display_result

from src.utils.processing_pipelines import process

import src.tools.interests.format as interests_format

from src.utils.helper_functions_defined_by_user._functions_helper import (
    filter_list_by_regex,
    str_split_safe,
    get_stats_for_column,
)


# local imports
from src.tools.interests.monitoring.utils_interests_monitoring import (
    get_affinities_hit_ratio,
    get_correlation_between,
    affinity_correlations_above_threhsold,
    affinity_correlations_below_threhsold,
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Settings

# COMMAND ----------

# max size of any dataset in case of choosing 'limit' run option in the notebook params
# (chosen empirically, enough to show something but doesn't take long)
limited_data_size = 10_000

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Delta paths settings:

# COMMAND ----------

@dp.notebook_function('%datalakebundle.table.defaults.storage_name%')
def define_paths(storage_name):
    home_dir_path = f"abfss://solutions@{storage_name}.dfs.core.windows.net/interests_monitoring/"
    
    return {
        "output_interest_keywords_num_hits":  f"{home_dir_path}interest_keywords_num_hits.delta",
        "output_interest_keywords_share":  f"{home_dir_path}interest_keywords_share.delta",
        "output_affinities_hit_ratio": f"{home_dir_path}affinities_hit_ratio.delta",
        "output_interest_useful_keywords": f"{home_dir_path}interest_useful_keywords.delta",
        "output_affinities_correlation": f"{home_dir_path}affinities_correlation.delta",
        "output_interest_set_per_keyword": f"{home_dir_path}interest_set_per_keyword.delta",
        "output_common_keywords_matrix": f"{home_dir_path}common_keywords_matrix.delta",
        "output_interest_common_keywords": f"{home_dir_path}interest_common_keywords.delta",
    }

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Daipe feature store:

# COMMAND ----------

user_entity = dp.fs.get_entity()
feature = dp.fs.feature_decorator_factory.create(user_entity)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Widgets

# COMMAND ----------

@dp.notebook_function()
def remove_all_widgets(widgets: dp.Widgets):
    widgets.remove_all()

# COMMAND ----------

@dp.notebook_function()
def create_widgets(widgets: dp.Widgets):
    # limit data size
    widgets.add_select(name="run_option",
                       choices=["limit", "full"],
                       default_value="limit",
                       label="Run option")
    
    # thresholds for affinities
    widgets.add_multiselect(name="affinities_thresholds",
                            choices=["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                            default_values=["0.0"],
                            label="Affinities thresholds")
    # correlation outliers thresholds 
    widgets.add_text(name="correlation_ub",
                     default_value="0.5",
                     label="Correlation upper bound")
    widgets.add_text(name="correlation_lb",
                     default_value="0.2",
                     label="Correlation lower bound")
    
    # time period
    widgets.add_text(name="fs_date",
                     default_value="", 
                     label="feature store date")
    widgets.add_text(name="pageview_n_days",
                     default_value="7",
                     label="n days for pageview")
    widgets.add_text(name="pageview_end_date",
                     default_value="",
                     label="end date for pageview")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Load inputs

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Load URL data:

# COMMAND ----------

@dp.transformation(dp.read_table("silver.sdm_url"), display=False)
def read_sdm_url(df: DataFrame):
    return df

# COMMAND ----------

@dp.transformation(dp.read_table("silver.sdm_pageview"), dp.get_widget_value("run_option"), dp.get_widget_value("pageview_end_date"), dp.get_widget_value("pageview_n_days"), display=False)
def read_sdm_pageview(df: DataFrame, run_option, end_date, n_days, logger: Logger):
    # get time period
    try:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    except:
        logger.warning("End date not set - the most recent data loaded.")
        end_date = date.today()
        
   
    start_date = end_date - timedelta(days=int(n_days))
    logger.info(f"Pageview data time interval: [{start_date}, {end_date}]")

    # load data
    df = df.filter((F.col("page_screen_view_date") >= start_date) & (F.col("page_screen_view_date") < end_date))
    return df.limit(limited_data_size) if run_option == "limit" else df

# COMMAND ----------

@dp.transformation(dp.read_table("silver.sdm_tokenized_domains"), dp.get_widget_value("run_option"), display=False)
def read_tokenized_domains(df: DataFrame, run_option):
    return df.limit(limited_data_size) if run_option == "limit" else df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Load interest defintions:

# COMMAND ----------

@dp.transformation(dp.read_delta('%interests.delta_path%'), '%processing.options.use_stemming%', '%processing.options.use_bigrams%', display=False)
def read_interests(df: DataFrame, use_stemming: bool, use_bigrams: bool, logger: Logger):
    processing_strategy = process.create_processing_strategy(use_stemming=use_stemming, 
                                                             use_bigrams=use_bigrams,
                                                             logger=logger)
    df_processed = process.process_interest_definitions(df,
                                                        input_col_single="keywords",
                                                        input_col_bigrams="keywords_bigrams",
                                                        processing_strategy=processing_strategy, 
                                                        logger=logger)
    
    return df_processed

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Extract interest names:

# COMMAND ----------

@dp.notebook_function(read_interests)
def get_interest_feature_names(df: DataFrame):
    return interests_format.subinterest_feature_names(df)

@dp.notebook_function(read_interests)
def get_affinity_feature_names(df: DataFrame):
    return interests_format.affinity_feature_names(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Load feature store:

# COMMAND ----------

@dp.transformation(user_entity, get_affinity_feature_names, dp.get_widget_value("fs_date"), dp.get_widget_value("run_option"), display=False)
def read_fs(entity, features_to_load: list, fs_date, run_option, feature_store: FeatureStoreTimestampGetter, logger: Logger):
    try:
        fs_date = datetime.strptime(fs_date, "%Y-%m-%d").date()
    except:
        fs_date = date.today()
        logger.warning("Feature store date not set - the most recent data loaded.")
    
    logger.info(f"Feature store date: {fs_date}")

    df = feature_store.get_for_timestamp(entity_name=entity.name, features=features_to_load, timestamp=fs_date, skip_incomplete_rows=True)

    return df.limit(limited_data_size) if run_option == "limit" else df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Combine URL and pageview:

# COMMAND ----------

@dp.transformation(read_sdm_pageview, read_sdm_url, display=False)
def combine_sdm(df_pageview: DataFrame, df_url: DataFrame):
    # TODO: this part needs to be refactored along with the main pipeline (it contains obsolete column names
    df_url = df_url.select("URL_NORMALIZED",
                           F.col("URL_TOKENS_ALL_CLEANED_UNIQUE").alias("TOKENS"),
                           F.col("URL_TOKENS_ALL_CLEANED_UNIQUE_BIGRAMS").alias("TOKENS_BIGRAMS"))
        
    return (df_pageview
            .select("user_id", "URL_NORMALIZED")
            .join(df_url, how="left", on="URL_NORMALIZED")
           )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##  Monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Tokenized domains
# MAGIC 
# MAGIC Interests - tokens based monitoring (monitoring of token hits for each interest).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Interest most frequent keywords
# MAGIC 
# MAGIC Most frequently hit keywords per interest

# COMMAND ----------

@dp.transformation(read_tokenized_domains, read_interests, get_interest_feature_names, display=False)
@display_result()
def get_interest_keywords_num_hits(df_tokenized_domains: DataFrame, df_interests, interest_names, logger: Logger):
    # get number of occurences for each kw
    df_tokens_cnt = df_tokenized_domains.groupBy("TOKEN").count()
    # combine <token, cnt> pair with all subinterests the token is part of
    df_tokens_cnt = (df_tokens_cnt
                     .withColumn("subinterest", F.explode(F.array(*map(F.lit, interest_names))))
                     .join(df_interests.select("subinterest", "keywords"), "subinterest")
                     .filter(F.array_contains(F.col("keywords"), F.col("TOKEN")))
                     .drop("keywords")
                    )
    if df_tokens_cnt.rdd.isEmpty():
        logger.warning("No tokens available (might be interest mismatch when joining the DFs).")
        return None
    # create ordering by number of occurences (per each subinterest)
    window = Window.partitionBy("subinterest").orderBy(F.col("count").desc())
    df_tokens_cnt = df_tokens_cnt.withColumn("row_num", F.row_number().over(window))
    # pivot the table to get a matrix (ordering X subinterest_name)
    df_tokens_cnt = df_tokens_cnt.toPandas()
    
    df_tokens_cnt["kv_pair"] = df_tokens_cnt.apply(lambda x: f'{x["TOKEN"]}: {x["count"]}', axis=1)
    df_pivot = df_tokens_cnt.pivot(index="row_num", columns="subinterest", values="kv_pair")

    return df_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Saving result to azure storage:

# COMMAND ----------

@dp.transformation(get_interest_keywords_num_hits, dp.get_widget_value("run_option"), display=False)
@dp.delta_append(define_paths.result['output_interest_keywords_num_hits'], options={'delta.autoOptimize.optimizeWrite': True, 'mergeSchema': True})
def save_interest_keywords_num_hits(df: DataFrame, run_option):
    df_spark = spark.createDataFrame(df)
    return (df_spark
            .select(
                F.lit(F.current_date()).alias('run_date'),
                F.lit(run_option).alias('run_option'),
                *df.columns,
            )
           )

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Keyword's share of hits
# MAGIC 
# MAGIC For each keyword, compute the share of its hits with respect to the whole interest (out of all interest hits, how many were caused by the keyword). This can serve as an estimate of how much a hit ratio of an interest would change if a particular keyword was omitted from the definition.
# MAGIC 
# MAGIC **Note**: The share computation is simplified as it doesn't count with the fact that there are often multiple keywords hit together by the same user. It rather estimates keyword's share by simply looking at the ratio of `number of hits of the keyword` / `total number of hits of the interest`. So, in extreme scenarios, an interest with an omitted keyword can actually have the same hit ratio as the original one, even though the keyword is pretty common (in case the keyword never appears alone). With that being said, it's not very likely to happen. 

# COMMAND ----------

@dp.transformation(get_interest_keywords_num_hits, display=False)
@display_result()
def get_interest_keywords_share(df_num_hits: DataFrame):
    df_split = pd.DataFrame()
    dict_sum_hits = {}
    for col_name, col_series in df_num_hits.iteritems():
        # split the keyword name & its frequency into separate columns
        df_split[f"{col_name}_keyword"] = col_series.apply(lambda x: str_split_safe(str(x), ":", position=0))
        df_split[f"{col_name}_count"] = col_series.apply(lambda x: str_split_safe(str(x), " ", position=1))
        # keep total number of hits
        dict_sum_hits[col_name] = df_split[f"{col_name}_count"].astype("float64").sum()
    
    for col_name in df_num_hits.columns: 
        # compute ratio (share) for each keyword
        df_split[f"{col_name}_share"] = (df_split
                                         .loc[:, f"{col_name}_count"]
                                         .astype("float64")
                                         .apply(lambda x: x if x is None else x / dict_sum_hits[col_name]) # pylint: disable = cell-var-from-loop
                                        )
    
        # collect back the keyword & its share into 1 common column
        df_split[col_name] = df_split.apply(lambda x: f'{x[f"{col_name}_keyword"]}: {x[f"{col_name}_share"]}', axis=1) # pylint: disable = cell-var-from-loop
        # drop temporary columns
        df_split.drop([f"{col_name}_keyword", f"{col_name}_count", f"{col_name}_share"], axis=1, inplace=True)
        df_split[col_name].replace({"nan: nan": None}, inplace=True)
    
    return df_split

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Saving result to azure storage:

# COMMAND ----------

@dp.transformation(get_interest_keywords_share, dp.get_widget_value("run_option"), display=False)
@dp.delta_append(define_paths.result['output_interest_keywords_share'], options={'delta.autoOptimize.optimizeWrite': True, 'mergeSchema': True})
def save_interest_keywords_share(df: DataFrame, run_option):
    df_spark = spark.createDataFrame(df)
    return (df_spark
            .select(
                F.lit(F.current_date()).alias('run_date'),
                F.lit(run_option).alias('run_option'),
                *df.columns,
            )
           )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Affinities hit ratio
# MAGIC 
# MAGIC Finding ratio of cases when affinities to each interest are positive / greater than threshold.

# COMMAND ----------

@dp.transformation(read_fs, dp.get_widget_value("affinities_thresholds"), display=False)
@display_result()
def all_affinities_hit_ratio(df_affinities, thresholds):
    df_aff_hit_ratio = get_affinities_hit_ratio(df_affinities, thresholds, spark)

    df_pd = df_aff_hit_ratio.toPandas()
    # prepare for pivoting (melt keeping only the threshold column)
    df_melted = pd.melt(df_pd, value_vars=[col for col in df_aff_hit_ratio.columns if col != 'threshold'], id_vars=['threshold'])
    df_melted["threshold"] = df_melted["threshold"].astype("string")
    # pivot to matrix: (variable x threshold)
    df_pivot = pd.pivot_table(df_melted, values='value', index=['threshold'], columns=['variable'], aggfunc='first').reset_index()

    return df_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Saving result to azure storage:

# COMMAND ----------

@dp.transformation(all_affinities_hit_ratio, dp.get_widget_value("run_option"), display=False)
@dp.delta_append(define_paths.result['output_affinities_hit_ratio'], options={'delta.autoOptimize.optimizeWrite': True, 'mergeSchema': True})
def save_affinities_hit_ratio(df: DataFrame, run_option):
    df_spark = spark.createDataFrame(df)
    return (df_spark
            .select(
                F.lit(F.current_date()).alias('run_date'),
                F.lit(run_option).alias('run_option'),
                *df.columns,
            )
           )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Useless interest keywords
# MAGIC 
# MAGIC Identifying tokens which do not appear in the tokenized domains table at all.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Useless / useful keyword sets
# MAGIC 
# MAGIC Separation of useless / useful keywords per interest:

# COMMAND ----------

@dp.transformation(read_tokenized_domains, read_interests, display=False)
def get_interest_useful_keywords(df_tokenized_domains: DataFrame, df_interests: DataFrame):
    # collect all tokens
    tokens_set = df_tokenized_domains.agg(F.collect_set(F.col("TOKEN"))).first()[0]
    # Find all useless keywords: explode interest keywords -> flag their presence in the tokens set
    # -> collect back missing keywords (do the oposite for useful)
    return (df_interests
            .withColumn("kw", F.explode(F.col("keywords")))
            .withColumn("tokens", F.array(*map(F.lit, tokens_set)))
            .withColumn("is_keyword_present", F.array_contains(F.col("tokens"), F.col("kw")))
            .withColumn("kw_useless", F.when(~F.col("is_keyword_present"), F.col("kw")))
            .withColumn("kw_useful", F.when(F.col("is_keyword_present"), F.col("kw")))
            .groupBy("subinterest")
            .agg(F.collect_set("kw_useless").alias("keywords_useless"),
                 F.collect_set("kw_useful").alias("keywords_useful"))
           )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Ratio of useless keywords per interest

# COMMAND ----------

@dp.transformation(get_interest_useful_keywords, display=False)
@display_result()
def get_interest_useful_keywords_ratio(df_usefulness: DataFrame):
    df_usefulness = (df_usefulness
                     .withColumn("keywords_useless_count", F.size("keywords_useless"))
                     .withColumn("keywords_useful_count", F.size("keywords_useful"))
                     .withColumn("useful_keywords_ratio", F.col("keywords_useful_count") / (F.col("keywords_useful_count") + F.col("keywords_useless_count")))
                     .sort(F.col("useful_keywords_ratio"))
                    )

    return df_usefulness

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Saving result to azure storage:

# COMMAND ----------

@dp.transformation(get_interest_useful_keywords_ratio, dp.get_widget_value("run_option"), display=False)
@dp.delta_append(define_paths.result['output_interest_useful_keywords'], options={'delta.autoOptimize.optimizeWrite': True, 'mergeSchema': True})
def save_interest_useful_keywords(df: DataFrame, run_option):
    return (df
            .select(
                F.lit(F.current_date()).alias('run_date'),
                F.lit(run_option).alias('run_option'),
                *df.columns,
            )
           )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Average usefulness:

# COMMAND ----------

@dp.transformation(get_interest_useful_keywords_ratio)
@display_result()
def display_useful_keywords_ratio_stats(df: DataFrame):
    df = get_stats_for_column(df=df, 
                              col="useful_keywords_ratio", 
                              n_steps_percentile=10)
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Cross-interest monitoring
# MAGIC 
# MAGIC Monitoring of interest uniqueness with respect to other interests.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Interest affinities correlaction
# MAGIC 
# MAGIC **Note**: When there are no data available, a dummy matrix (with zeros everywhere is used instead).

# COMMAND ----------

@dp.transformation(read_fs, display=False)
def get_features_correlation(df_affinities: DataFrame, logger: Logger):
    # assemble all affinity features into a single column
    aff_col_prefix = "ad_interest_affinity_"
    cols_affinities = filter_list_by_regex(df_affinities.columns,
                                           regex=f"^{aff_col_prefix}.*")
    n_cols = len(cols_affinities)
    df_features = VectorAssembler(inputCols=cols_affinities, outputCol='features').transform(df_affinities)
    
    # if there was no data, the correlation matrix creation would throw an exception
    if df_features.count() == 0:
        logger.warning("cannot create the correlation matrix - no features are available.")
        n_cols = len(cols_affinities)
        data_dummy = [[0.0] * n_cols for _ in range(n_cols)]
        return pd.DataFrame(data_dummy, index=cols_affinities, columns=cols_affinities)
    
    # compute correlation matrix & extract it from the dataframe
    df_matrix = Correlation.corr(df_features, 'features').first()[0]
    # convert DenseMatrix to pandas DF with readable column names
    prefix_len = len(aff_col_prefix)
    cols_affinities = [c[prefix_len:].upper() for c in cols_affinities]

    return pd.DataFrame(df_matrix.values.reshape(-1, n_cols), index=cols_affinities, columns=cols_affinities)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Saving result to azure storage:

# COMMAND ----------

@dp.transformation(get_features_correlation, dp.get_widget_value("run_option"), display=False)
@dp.delta_append(define_paths.result['output_affinities_correlation'], options={'delta.autoOptimize.optimizeWrite': True, 'mergeSchema': True})
def save_affinities_correlation(df: DataFrame, run_option):
    df_spark = spark.createDataFrame(df.reset_index())
    return (df_spark
            .select(
                F.lit(F.current_date()).alias('run_date'),
                F.lit(run_option).alias('run_option'),
                *df.columns,
            )
           )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Correlation matrix
# MAGIC 
# MAGIC Display the correlation matrix using [seaborn heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html):

# COMMAND ----------

@dp.notebook_function(get_features_correlation)
def display_corr_matrix(df: DataFrame):
    # define cmap & change plot size to make stuff readable
    cmap = sns.diverging_palette(300, 10, as_cmap=True)
    _ = plt.subplots(figsize=(25, 25))
    # render the heatmap
    _ = sns.heatmap(df,
                    cmap=cmap,
                    vmax=1,
                    center=0,
                    square=True,
                    linewidths=.5,
                    cbar_kws={
                        "shrink": .5,
                        "fraction": 0.05,
                    }
                   )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Correlation thresholds
# MAGIC 
# MAGIC Displaying interests with affinity-correlations above/bellow pre-defined thresholds

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Columns with correlation above a threshold:

# COMMAND ----------

@dp.notebook_function(get_features_correlation, dp.get_widget_value("correlation_ub"))
def corr_above_threshold(df: DataFrame, correlation_ub, logger: Logger):
    df = df.copy(deep=True)
    df['index_column'] = df.index

    # above threshold
    items_above_threshold = affinity_correlations_above_threhsold(df_corr=df, threshold=float(correlation_ub))
    # print all correlations over the threshold
    any_output = False
    for index, value in items_above_threshold.iteritems():
        if len(value) > 0:
            print(f"{index}:")
            # sort by correlation value
            correlations_sorted = list((get_correlation_between(df_corr=df, first=index, second=column_name), column_name) for column_name in value)
            correlations_sorted.sort(key=lambda x: x[0])
            for corr_value, column_name in correlations_sorted:
                print(f"  {corr_value:.4f}: {column_name}")
                any_output = True
    if not any_output: 
        logger.info("There are no interests with mutual correlation above given threshold.")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Columns with most correlation coefficient bellow a threshold:

# COMMAND ----------

@dp.notebook_function(get_features_correlation, dp.get_widget_value("correlation_lb"))
def corr_below_threshold(df: DataFrame, correlation_lb, logger: Logger):
    df = df.copy(deep=True)
    df['index_column'] = df.index

    items_below_threshold = affinity_correlations_below_threhsold(df_corr=df, threshold=float(correlation_lb))
    any_output = False
    for index, row in items_below_threshold.iterrows():
        print(f"{row['len_correlated']}: {index}")
        any_output = True
    if not any_output: 
        logger.info("There are no interests with mutual correlation below given threshold.")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Common vs unique keywords
# MAGIC 
# MAGIC Finding unique keywords in interest definitions

# COMMAND ----------

@dp.transformation(read_interests, display=False)
def get_interest_common_keywords(df_interests: DataFrame):
    # group by keywords instead of interests, then group again by collected set of interests
    # to see what keywords do interests in the set have in common. Exploding both sets again
    # & collecting all keywords for each interests leaves with set of non-unique keywords
    return (df_interests
            .withColumn("kw", F.explode("keywords"))
            .groupby("kw")
            .agg(F.collect_set("subinterest").alias("subinterest_set"))
            .filter(F.size("subinterest_set") > 1)
            .groupby("subinterest_set")
            .agg(F.collect_set("kw").alias("keywords"))
            .withColumn("subinterest", F.explode("subinterest_set"))
            .withColumn("kw", F.explode("keywords"))
            .groupby("subinterest")
            .agg(F.collect_set("kw").alias("common_keywords"))
           )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### All interest per keyword (saving)
# MAGIC 
# MAGIC Displaying a list of interests for a given keyword

# COMMAND ----------

@dp.transformation(read_interests, display=False)
def get_interest_set_per_keyword(df_interests: DataFrame):
    df = (df_interests
          .withColumn("keyword", F.explode("keywords"))
          .groupBy("keyword")
          .agg(F.collect_set("subinterest").alias("subinterests"))
         )

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Saving result to azure storage:

# COMMAND ----------

@dp.transformation(get_interest_set_per_keyword, dp.get_widget_value("run_option"), display=False)
@dp.delta_append(define_paths.result['output_interest_set_per_keyword'], options={'delta.autoOptimize.optimizeWrite': True, 'mergeSchema': True})
def save_interest_set_per_keyword(df: DataFrame, run_option):
    return (df
            .select(
                F.lit(F.current_date()).alias('run_date'),
                F.lit(run_option).alias('run_option'),
                *df.columns,
            )
           )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Common keywords for 2 given interests (saving):
# MAGIC 
# MAGIC Get matrix of (interests X interests) with values being their shared keywords

# COMMAND ----------

@dp.transformation(get_interest_common_keywords, get_interest_feature_names, display=False)
def get_interest_common_keywords_matrix(df_common_kw: DataFrame, interest_names: list):
    # prepare copy of the dataset for a join
    df_other = df_common_kw.withColumnRenamed("common_keywords", "common_keywords_other")
    # create all possible interest pairs (interest X interest_other)
    df_common_kw =  (df_common_kw
                     .withColumn("subinterest", F.upper("subinterest"))
                     .withColumn("all_interests", F.array(*map(F.lit, interest_names)))
                     .withColumn("subinterest_other", F.explode("all_interests"))
                     .withColumn("subinterest_other", F.upper("subinterest_other"))
                     .filter(F.col("subinterest") != F.col("subinterest_other"))
                    )
    # join common keywords set for the 'interest_other' column
    df_common_kw = (df_common_kw
                    .join(df_other, df_common_kw["subinterest_other"] == F.upper(df_other["subinterest"]), "left")
                    .drop(df_other["subinterest"])
                   )
    # keep only the intersection between both sets (keywords X keywords_other)
    df_common_kw = (df_common_kw
                    .withColumn("kw", F.explode("common_keywords"))
                    .withColumn("kw_other", F.explode("common_keywords_other"))
                    .withColumn("kw_equal", F.when(F.col("kw") == F.col("kw_other"), F.col("kw")))
                    .groupby(["subinterest", "subinterest_other"])
                    .agg(F.collect_set("kw_equal").alias("keywords_common"))
                   )
    df_common_kw_pivot = df_common_kw.toPandas().pivot(index="subinterest",
                                                       columns="subinterest_other",
                                                       values="keywords_common")

    return df_common_kw_pivot.reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Saving result to azure storage:

# COMMAND ----------

@dp.transformation(get_interest_common_keywords_matrix, dp.get_widget_value("run_option"), display=False)
@dp.delta_append(define_paths.result['output_common_keywords_matrix'], options={'delta.autoOptimize.optimizeWrite': True, 'mergeSchema': True})
def save_common_keywords_matrix(df: DataFrame, run_option):
    df_spark = spark.createDataFrame(df)
    return (df_spark
            .select(
                F.lit(F.current_date()).alias('run_date'),
                F.lit(run_option).alias('run_option'),
                *df.columns,
            )
           )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Separating sets of unique & non-unique keywords
# MAGIC 
# MAGIC Splitting the interest definition (keywords) into two disjunct sets - keywords unique for the interest / the rest

# COMMAND ----------

@dp.transformation(read_interests, get_interest_common_keywords, display=False)
@display_result()
def get_interest_unique_keywords_ratio(df_interests: DataFrame, df_common_kws: DataFrame):
    # join with original interests dataframe, subtract <common> set from from <all> set to get <unique> set
    df_unique_interests = (df_interests
                           .join(df_common_kws, "subinterest")
                           .withColumn("kw", F.explode("keywords"))
                           .withColumn("kw_unique", F.when(~F.array_contains(F.col("common_keywords"), F.col("kw")), F.col("kw")))
                           .groupby("subinterest")
                           .agg(F.collect_set("kw_unique").alias("keywords_unique"))
                          )
    # add to the original DFs
    df_interests = (df_interests
                    .join(df_unique_interests, "subinterest")
                    .join(df_common_kws, "subinterest")
                   )
    # sort by uniqueness
    df_interests = (df_interests
                    .withColumn("keywords_unique_count", F.size("keywords_unique"))
                    .withColumn("keywords_common_count", F.size("common_keywords"))
                    .withColumn("keywords_unique_ratio", F.col("keywords_unique_count") / (F.col("keywords_unique_count") + F.col("keywords_common_count")))
                    .sort(F.col("keywords_unique_ratio"))
                    .select(
                        "subinterest",
                        "general_interest",
                        "keywords_unique_count",
                        "keywords_common_count",
                        "keywords_unique_ratio",
                        "keywords",
                        "keywords_unique",
                        "common_keywords",
                    )
                   )
    
    return df_interests

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Saving result to azure storage:

# COMMAND ----------

@dp.transformation(get_interest_unique_keywords_ratio, dp.get_widget_value("run_option"), display=False)
@dp.delta_append(define_paths.result['output_interest_common_keywords'], options={'delta.autoOptimize.optimizeWrite': True, 'mergeSchema': True})
def save_interest_common_keywords(df: DataFrame, run_option):
    return (df
            .select(
                F.lit(F.current_date()).alias('run_date'),
                F.lit(run_option).alias('run_option'),
                *df.columns,
            )
           )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Data monitoring
# MAGIC 
# MAGIC Purely data-based monitoring (does not use definitions of interests at all).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Token absolute frequency
# MAGIC 
# MAGIC Most frequent tokens in total.

# COMMAND ----------

@dp.transformation(combine_sdm, display=False)
@display_result()
def get_tokens_frequency(df: DataFrame):
    df_agg = (df
              .withColumn("TOKEN", F.explode("TOKENS"))
              .groupBy("TOKEN")
              .count()
              .orderBy(F.col("count").desc())
             )

    return df_agg

# COMMAND ----------

@dp.notebook_function(read_interests)
def get_all_keywords_used(df_interests):
    # return list of all unique keywords across interests
    df_all = (df_interests
              .withColumn("kw", F.explode("keywords"))
              .agg(F.collect_set("kw"))
             )
    
    return df_all.first()[0]

# COMMAND ----------

@dp.transformation(get_tokens_frequency, get_all_keywords_used, display=False)
def get_token_usage(df_tokens: DataFrame, keywords_used):
    # flag tokens when they appear in any of the interest definitions
    return (df_tokens
            .withColumn("is_used", F.array_contains(F.array(*map(F.lit, keywords_used)),
                                                    F.col("TOKEN"))
                       )
           )

# COMMAND ----------

@dp.transformation(get_token_usage, display=False)
@display_result()
def display_tokens_used(df_usage: DataFrame):
    df = (df_usage
          .filter(F.col("is_used"))
          .drop("is_used")
         )
    
    return df

# COMMAND ----------

@dp.transformation(get_token_usage, display=False)
@display_result()
def display_tokens_unused(df_usage: DataFrame):
    df = (df_usage
          .filter(~F.col("is_used"))
          .drop("is_used")
         )
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Bigrams absolute frequency
# MAGIC 
# MAGIC Most frequent bigrams in total.

# COMMAND ----------

@dp.transformation(combine_sdm, display=False)
@display_result()
def get_bigrams_frequency(df: DataFrame):
    df_agg = (df
            .withColumn("TOKEN", F.explode("TOKENS_BIGRAMS"))
            .groupBy("TOKEN")
            .count()
            .orderBy(F.col("count").desc())
           )

    return df_agg

# COMMAND ----------

@dp.notebook_function(read_interests)
def get_all_bigrams_used(df_interests):
    # return list of all unique keywords across interests
    df_all = (df_interests
              .withColumn("kw", F.explode("keywords_bigrams"))
              .agg(F.collect_set("kw"))
             )
    return df_all.first()[0]

# COMMAND ----------

@dp.transformation(get_bigrams_frequency, get_all_bigrams_used, display=False)
def get_bigram_usage(df_tokens: DataFrame, keywords_used):
    # flag tokens when they appear in any of the interest definitions
    return (df_tokens
            .withColumn("is_used", F.array_contains(F.array(*map(F.lit, keywords_used)),
                                                    F.col("TOKEN"))
                       )
           )

# COMMAND ----------

@dp.transformation(get_bigram_usage, display=False)
@display_result()
def display_bigrams_used(df_usage: DataFrame):
    df = (df_usage
          .filter(F.col("is_used"))
          .drop("is_used")
         )
   
    return df

# COMMAND ----------

@dp.transformation(get_bigram_usage, display=False)
@display_result()
def display_bigrams_unused(df_usage: DataFrame):
    df = (df_usage
          .filter(~F.col("is_used"))
          .drop("is_used")
          )
    
    return df
