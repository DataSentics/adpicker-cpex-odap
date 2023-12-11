# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # New interest definition
# MAGIC 
# MAGIC This notebook serves as a general tool for when defining keywords for new interests.
# MAGIC 
# MAGIC 
# MAGIC Notion page with recommended flow for new interest definition: [How to define or update interest](https://www.notion.so/datasentics/How-to-define-or-update-interest-689d7f0059aa4990b9b8371d7a169e5c).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The notebook does multiple tasks (so that they can run as one instance when needed). There is binary flag parameter for each use case (*True* / *False*) so that some use cases can be skipped when necessary. Tasks:
# MAGIC * **frequent keywords** (*display_frequent*): looking at the most frequent tokens / bigrams in the data (possibly filtered only to certain web pages).
# MAGIC * **suggestions** (*display_suggestions*): using current definitions to recommend new keywords.
# MAGIC * analysis of current definition
# MAGIC   - **conflicts** (*display_conflicts*): conflicting keywords with other (already existing) interests.
# MAGIC   - **monitoring**: monitoring of new interest's hits. Separated into:
# MAGIC     * **single-word monitoring** (*display_monitoring_single*): less time consuming, it is possible to use much larger dataset
# MAGIC     * **bigram monitoring** (*display_monitoring_bigram*): much more time consuming, requires smaller input data

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

import pandas as pd
from logging import Logger

import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.feature import Word2VecModel 


# project level imports
from src.utils.helper_functions_defined_by_user._stop_words import unwanted_tokens
from src.utils.helper_functions_defined_by_user._functions_nlp import strip_diacritic
from src.utils.helper_functions_defined_by_user._functions_udf import udf_reverse_word_order
from src.utils.helper_functions_defined_by_user._functions_helper import (
    add_binary_flag_widget,
    check_binary_flag_widget, 
    get_top_n_sorted,
    flatten_string_iterable,
    format_fraction,
)

from src.utils.processing_pipelines import process
import src.tools.interests.format as interests_format

# TODO this should not be needed
from adpickercpex.lib.display_result import display_result


# local imports
from src.tools.interests.define_update_interest.interest_definitions import (
    InterestDefinition, 
    get_interest_definition,
)
from src.tools.interests.define_update_interest.utils_interest_definition import (
    get_most_frequent_per_column, 
    jlh,
    url_hits_for_group,
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Widgets & settings

# COMMAND ----------

@dp.notebook_function()
def remove_all_widgets(widgets: dp.Widgets):
    widgets.remove_all()

# COMMAND ----------

@dp.notebook_function()
def create_widgets(widgets: dp.Widgets):
    # allow to limit data size
    widgets.add_select(name="run_option",
                   choices=["limit", "full"],
                   default_value="limit",
                   label="Run option")
    # current interest definition to be examined  
    widgets.add_text(name="interest_name",
                   default_value="tmp",
                   label="Interest name")
    # prepared data sample tag  
    widgets.add_text(name="data_tag",
                   default_value="",
                   label="Data sample tag")
    
    # binay flags - enable/disable separate parts of the notebook (True / False)
    add_binary_flag_widget(widgets, name="display_frequent", label="Display frequent tokens / bigrams")
    add_binary_flag_widget(widgets, name="display_suggestions")
    add_binary_flag_widget(widgets, name="display_conflicts")
    add_binary_flag_widget(widgets, name="display_monitoring_single")
    add_binary_flag_widget(widgets, name="display_monitoring_bigram")
    add_binary_flag_widget(widgets, name="display_url_monitoring")
    
    # ** optional parameters ** 
    
    # url filtering regex
    widgets.add_text(name="url_regex",
                   default_value="(.*)",
                   label="URL regex")
    
    # limit all TOP-N outputs
    widgets.add_text(name="top_n_frequent",
                   default_value="100",
                   label="Top N frequent tokens/bigrams")
    widgets.add_text(name="top_n_w2v",
                   default_value="100",
                   label="Top N recommended W2V suggestions")
    widgets.add_text(name="top_n_jlh",
                   default_value="100",
                   label="Top N recommended JLH suggestions")
    widgets.add_text(name="top_n_url",
                   default_value="100",
                   label="Top N common URLs")

# COMMAND ----------

_get_widget_flag_bool = lambda x: check_binary_flag_widget(dbutils.widgets.get(x))


DISPLAY_FREQUENT = _get_widget_flag_bool("display_frequent")
DISPLAY_SUGGESTIONS = _get_widget_flag_bool("display_suggestions")
DISPLAY_CONFLICTS = _get_widget_flag_bool("display_conflicts")
DISPLAY_MONITORING_SINGLE = _get_widget_flag_bool("display_monitoring_single")
DISPLAY_MONITORING_BIGRAM = _get_widget_flag_bool("display_monitoring_bigram")
DISPLAY_URL_MONITORING = _get_widget_flag_bool("display_url_monitoring")

# COMMAND ----------

# max size of any dataset in case of choosing 'limit' run option in the notebook params
# (chosen empirically, enough to show something but doesn't take long)
limited_data_size = 10_000

# COMMAND ----------

@dp.notebook_function('%datalakebundle.table.defaults.storage_name%', dp.get_widget_value("data_tag"))
def define_paths(storage_name, data_tag, logger: Logger):
    home_dir_path = f"abfss://solutions@{storage_name}.dfs.core.windows.net/new_interest_definition/"
    def _define_paths_with_tag(tag: str): 
        return {
            'input_path_url': f'{home_dir_path}url_level_{tag}.delta',
            'input_path_user_id': f'{home_dir_path}user_id_level_{tag}.delta',
        }
    
    result = _define_paths_with_tag(data_tag)
    # check existance (valid tags), use default tag when some of the files don't exist
    for path in result.values():
        try:
            dbutils.fs.ls(path)
        except Exception:
            logger.warning(f"Invalid data tag used: `{data_tag}`. The default data sample paths are used instead.")
            return _define_paths_with_tag(tag="")
            
    logger.info(f"All paths for the data tag `{data_tag}` sucesfully defined.")
    
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Load inputs

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Load URL data:

# COMMAND ----------

@dp.transformation(dp.read_delta(define_paths.result['input_path_url']), dp.get_widget_value("run_option"), dp.get_widget_value("url_regex"), display=False)
def load_url_level(df: DataFrame, run_option, url_regex):
    # filter out unwanted URLs
    df = df.filter(F.col("URL_NORMALIZED").rlike(url_regex))
    # limit data size 
    return df.limit(limited_data_size) if run_option == "limit" else df

# COMMAND ----------

@dp.transformation(dp.read_delta(define_paths.result['input_path_user_id']), dp.get_widget_value("run_option"), display=False)
def load_user_id_level(df: DataFrame, run_option):
    # limit data size 
    return df.limit(limited_data_size) if run_option == "limit" else df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Load interest defintions:

# COMMAND ----------

@dp.notebook_function('%processing.options.use_stemming%', '%processing.options.use_bigrams%')
def get_processing_strategy(use_stemming: bool, use_bigrams: bool, logger: Logger):
    processing_strategy = process.create_processing_strategy(use_stemming=use_stemming, 
                                                             use_bigrams=use_bigrams, 
                                                             logger=logger)
    return processing_strategy

# COMMAND ----------

@dp.transformation(dp.read_delta('%interests.delta_path%'), get_processing_strategy, display=False)
def read_interests(df: DataFrame, processing_strategy, logger: Logger):
    df_processed = process.process_interest_definitions(df,
                                                        input_col_single="keywords",
                                                        input_col_bigrams="keywords_bigrams",
                                                        processing_strategy=processing_strategy, 
                                                        logger=logger)
    
    return df_processed

# COMMAND ----------

@dp.notebook_function(read_interests, get_processing_strategy)
def get_interest_definitions_to_dict(df: DataFrame, processing_strategy): 
    # pick formatting function based on the use of bigrams
    format_f = (interests_format.interest_definitions_to_dict_all 
                 if process.get_active_flag_bigrams(processing_strategy)
                 else interests_format.interest_definitions_to_dict_single)
    
    return format_f(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Focused tokens selection
# MAGIC 
# MAGIC Selection of relevant tokens (e.g. keywords for particular interest) to be used for further analysis.

# COMMAND ----------

@dp.notebook_function(dp.get_widget_value("interest_name"))
def read_interest_definition(interest_name): 
    return get_interest_definition(interest_name)

# COMMAND ----------

@dp.notebook_function(read_interest_definition)
def get_keywords_single_only(interest_definition: InterestDefinition):
    return interest_definition.single_tokens
    
@dp.notebook_function(read_interest_definition)
def get_keywords_bigrams_only(interest_definition: InterestDefinition):
    return interest_definition.bigrams

@dp.notebook_function(read_interest_definition)
def get_keywords_all(interest_definition: InterestDefinition): 
    return interest_definition.all_tokens

@dp.notebook_function(read_interest_definition)
def get_keywords_all_flat(interest_definition: InterestDefinition):
    return interest_definition.all_tokens_flat
  
@dp.notebook_function(read_interest_definition)
def get_keywords_bigrams_only_flat(interest_definition: InterestDefinition):
    return interest_definition.bigrams_flat

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## New keywords suggestions

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Displaying frequent keywords

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Single words:

# COMMAND ----------

@dp.transformation(load_url_level, dp.get_widget_value("top_n_frequent"), display=False)
def most_frequent_tokens(df: DataFrame, top_n_frequent):
    df_freq = get_most_frequent_per_column(df.select("TOKENS"), int(top_n_frequent))
    return df_freq

# COMMAND ----------

@dp.transformation(most_frequent_tokens, display=False)
@display_result(display=DISPLAY_FREQUENT)
def display_most_frequent_tokens(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Bigrams:

# COMMAND ----------

@dp.transformation(load_url_level, dp.get_widget_value("top_n_frequent"), display=False)
def most_frequent_bigrams(df: DataFrame, top_n_frequent):
    df_freq = get_most_frequent_per_column(df.select("BIGRAMS"), int(top_n_frequent))
    return df_freq

# COMMAND ----------

@dp.transformation(most_frequent_bigrams, display=False)
@display_result(display=DISPLAY_FREQUENT)
def display_most_frequent_bigrams(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Frequent neighbours
# MAGIC 
# MAGIC Finding tokens which appear the most together with the selected ones (absolute frequency).
# MAGIC 
# MAGIC 
# MAGIC For the purpose of this query, bigrams are split into separate words.

# COMMAND ----------

@dp.transformation(load_url_level, get_keywords_all_flat, dp.get_widget_value("top_n_frequent"), display=False)
def most_frequent_neighbours(df: DataFrame, focused_tokens, top_n_frequent, logger: Logger):
    if len(focused_tokens) == 0: 
        logger.warning("Skipping, no tokens available.")
        return None
    
    df_freq = (df
               .withColumn("token", F.explode("TOKENS"))
               .filter(F.array_contains(F.array(*map(F.lit, focused_tokens)), F.col("token")))
               .withColumn("neighbour", F.explode("TOKENS"))
               .filter(F.col("token") != F.col("neighbour"))
               .groupBy("neighbour")
               .count()
               .orderBy(F.col("count").desc())
               .limit(int(top_n_frequent))
              )
    
    return df_freq

# COMMAND ----------

@dp.transformation(most_frequent_neighbours, display=False)
@display_result(display=DISPLAY_FREQUENT)
def display_most_frequent_neighbours(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Word2Vec
# MAGIC 
# MAGIC Finding tokens with the most similar embedings.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Prepare tokens (bigrams are split into single keywords and combined with the rest):

# COMMAND ----------

@dp.notebook_function(get_keywords_all_flat)
def prepare_tokens_w2v(keywords):
    # flattening
    tokens = flatten_string_iterable(keywords)
    # get to lower-case, remove stop words & diacrtitics
    return [strip_diacritic(word).lower() 
            for word in tokens 
            if word not in unwanted_tokens]

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Load model:

# COMMAND ----------

@dp.notebook_function()
def load_model_w2v():
    return Word2VecModel.load("/mnt/models/word2vec_model/czech_web_1mil.model")

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Use the model to find similar words:

# COMMAND ----------

@dp.transformation(load_model_w2v, prepare_tokens_w2v, dp.get_widget_value("top_n_w2v"), display=False)
def top_n_similar_w2v(model_w2v, tokens, top_n_w2v, logger: Logger):
    top_n = int(top_n_w2v)
    # init empty dataframe
    col_word, col_score, col_src = "word", "similarity", "source"
    output_schema = f"{col_word} string, {col_score} double, {col_src} string"
    df_result = spark.createDataFrame(data=spark.sparkContext.emptyRDD(), 
                                      schema=output_schema) 
    # append similar words for each token to the result
    tokens_skipped = []
    for token in tokens: 
        try:
            df_token = model_w2v.findSynonyms(token, top_n)
            df_token = df_token.withColumn(col_src, F.lit(token))
            df_result = df_result.union(df_token)
        except Exception: 
            tokens_skipped.append(token)
            
    if len(tokens_skipped) > 0:
        logger.warning(f"Some tokens are missing in the W2V vocabulary: {tokens_skipped}")
    else:
        logger.info("All input tokens are in the W2V vocabulary.")
        
    # filter only to new tokens & drop duplicates
    df_filtered = (df_result
                   .filter(~F.col(col_word).isin(tokens))
                   .groupBy(col_word)
                   .agg(F.max(col_score).alias(col_score), 
                        F.first(col_src).alias(col_src))
                  )
    # display only tokens with the highest similarity score
    df_top = (df_filtered
              .orderBy(F.desc(col_score))
              .limit(top_n)
             )
    
    return df_top

# COMMAND ----------

@dp.transformation(top_n_similar_w2v, display=False)
@display_result(display=DISPLAY_SUGGESTIONS)
def display_top_n_similar_w2v(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### JLH score
# MAGIC 
# MAGIC Computing JLH score for both single words and bigrams. Token sets are flagged positive if they lead to positive affinity (at least 1 keyword or bigram match).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Preparations

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Flagging separate rows by positive affinities based on tokens / bigrams:

# COMMAND ----------

@dp.transformation(load_user_id_level, get_keywords_single_only, display=False)
def prepare_jhl_flags_single(df: DataFrame, keywords):
    df_flags = (df
                .withColumn('DEFINITION', F.array(*map(F.lit, keywords)))
                .withColumn('LABEL', F.when(F.size(F.array_intersect('TOKENS', 'DEFINITION')) > 0, 1).otherwise(0))
               )
    
    return df_flags

@dp.transformation(load_user_id_level, get_keywords_bigrams_only, display=False)
def prepare_jhl_flags_bigram(df: DataFrame, keywords):
    df_flags = (df
          .withColumn('DEFINITION', F.array(*map(F.lit, keywords)))
          .withColumn('LABEL', F.when(F.size(F.array_intersect('BIGRAMS', 'DEFINITION')) > 0, 1).otherwise(0))
         )
    
    return df_flags

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Computing the JLH score for both single words and bigrams flagged by both as well:

# COMMAND ----------

@dp.transformation(prepare_jhl_flags_single, display=False)
def single_jlh_score_single_flags(df: DataFrame):
    return jlh(df=df, data_col="TOKENS", doc_id_col="USER_ID")

@dp.transformation(prepare_jhl_flags_bigram, display=False)
def single_jlh_score_bigram_flags(df: DataFrame):
    return jlh(df=df, data_col="TOKENS", doc_id_col="USER_ID")

@dp.transformation(prepare_jhl_flags_single, display=False)
def bigram_jlh_score_single_flags(df: DataFrame):
    return jlh(df=df, data_col="BIGRAMS", doc_id_col="USER_ID")

@dp.transformation(prepare_jhl_flags_bigram, display=False)
def bigram_jlh_score_bigram_flags(df: DataFrame):
    return jlh(df=df, data_col="BIGRAMS", doc_id_col="USER_ID")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Single words recommendation
# MAGIC 
# MAGIC Recommending new tokens based on both flag origins.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Top scored words flagged by single words:

# COMMAND ----------

@dp.transformation(single_jlh_score_single_flags, get_keywords_single_only, dp.get_widget_value("top_n_jlh"), display=False)
@display_result(display=DISPLAY_SUGGESTIONS)
def display_single_jlh_score_single_flags_top(df: DataFrame, definition_keywords, top_n_jlh):
    df_filtered = (df
                   .filter(~F.col("token").isin(definition_keywords))
                  )
    df_top = get_top_n_sorted(df_filtered, 
                              column_name="JLH_score", 
                              n=int(top_n_jlh))
    
    return df_top

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Top scored words flagged by bigrams:

# COMMAND ----------

@dp.transformation(single_jlh_score_bigram_flags, get_keywords_single_only, dp.get_widget_value("top_n_jlh"), display=False)
@display_result(display=DISPLAY_SUGGESTIONS)
def display_single_jlh_score_bigram_flags_top(df: DataFrame, definition_keywords, top_n_jlh):
    df_filtered = (df
                   .filter(~F.col("token").isin(definition_keywords))
                  )
    df_top = get_top_n_sorted(df_filtered, 
                              column_name="JLH_score", 
                              n=int(top_n_jlh))
    
    return df_top

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Bigrams words recommendation
# MAGIC 
# MAGIC Recommending new bigrams based on both flag origins.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Top scored bigrams flagged by single words:

# COMMAND ----------

@dp.transformation(bigram_jlh_score_single_flags, get_keywords_bigrams_only, dp.get_widget_value("top_n_jlh"), display=False)
@display_result(display=DISPLAY_SUGGESTIONS)
def display_bigram_jlh_score_single_flags_top(df: DataFrame, definition_keywords, top_n_jlh):
    df_filtered = (df
                   .filter(~F.col("token").isin(definition_keywords))
                  )
    df_top = get_top_n_sorted(df_filtered, 
                              column_name="JLH_score", 
                              n=int(top_n_jlh))
    
    return df_top

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Top scored bigrams flagged by bigrams:

# COMMAND ----------

@dp.transformation(bigram_jlh_score_bigram_flags, get_keywords_bigrams_only, dp.get_widget_value("top_n_jlh"), display=False)
@display_result(display=DISPLAY_SUGGESTIONS)
def bigram_jlh_score_bigram_flags_top(df: DataFrame, definition_keywords, top_n_jlh):
    df_filtered = (df
                   .filter(~F.col("token").isin(definition_keywords))
                  )
    df_top = get_top_n_sorted(df_filtered, 
                              column_name="JLH_score", 
                              n=int(top_n_jlh))
    
    return df_top

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Conflicts with other interests
# MAGIC 
# MAGIC Displaying all keyword conflicts with already defined interests

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC By keyword:

# COMMAND ----------

@dp.transformation(read_interests, get_keywords_all, display=False)
def keyword_conflicts_by_kw(df_interests: DataFrame, tokens):
    df = (df_interests
          .withColumn("kw", F.explode("keywords"))
          .filter(F.array_contains(F.array(*map(F.lit, tokens)), F.col("kw")))
          .groupBy("kw")
          .agg(F.collect_set("subinterest").alias("subinterest_set"))
          .withColumn("n_conlficts", F.size("subinterest_set"))
          .orderBy(F.desc("n_conlficts"))
         )
    
    return df

# COMMAND ----------

@dp.transformation(keyword_conflicts_by_kw, display=False)
@display_result(display=DISPLAY_CONFLICTS)
def display_keyword_conflicts_by_kw(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC By interest:

# COMMAND ----------

@dp.transformation(keyword_conflicts_by_kw, display=False)
def keyword_conflicts_by_interest(df: DataFrame):    
    df = (df
          .withColumn("subinterest", F.explode("subinterest_set"))
          .groupBy("subinterest")
          .agg(F.collect_set("kw").alias("keywords"))
          .withColumn("n_conlficts", F.size("keywords"))
          .orderBy(F.desc("n_conlficts"))
         )
    
    return df

# COMMAND ----------

@dp.transformation(keyword_conflicts_by_interest, display=False)
@display_result(display=DISPLAY_CONFLICTS)
def display_keyword_conflicts_by_interest(df: DataFrame):
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Interest definition monitoring
# MAGIC 
# MAGIC Monitoring of based on current definiton of the interest.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Number of hits
# MAGIC 
# MAGIC Monitoring absolute number of occurences of each word / bigram that is in the definition.

# COMMAND ----------

# number of decimals to round to when displaying
RATIO_PRECISION = 5

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Single keywords
# MAGIC 
# MAGIC Displaying number of hits of each keyword:

# COMMAND ----------

@dp.transformation(load_url_level, get_keywords_single_only, display=False)
def prepare_url_single_keyword_hits(df: DataFrame, keywords):
    # count number of hits for each token
    df_hits = (df 
               .withColumn("token", F.explode("TOKENS"))
               .filter(F.array_contains(F.array(*map(F.lit, keywords)), F.col("token")))
              )
    
    return df_hits

# COMMAND ----------

@dp.transformation(prepare_url_single_keyword_hits, get_keywords_single_only, display=False)
def hits_per_single_keyword(df: DataFrame, keywords, logger: Logger):
    if len(keywords) == 0: 
        logger.warning("Skipping: there are no keywords in the interest definition.")
        return None
        
    # prepare all keywords as a DF
    df_keywords = spark.createDataFrame(data=[Row(x) for x in keywords], schema=["token"])
    # get hits count
    df_hits = (df
               .groupBy("token")
               .count()
               .join(df_keywords, "token", "right")
               .fillna(0)
               .orderBy(F.desc("count"))
              )
    
    return df_hits

# COMMAND ----------

@dp.transformation(hits_per_single_keyword, dp.get_widget_value("display_monitoring_single"), display=False)
def hits_per_single_keyword_ratio(df: DataFrame, display):
    if df is None:
        return None
    # add percentage column (possibly dummy, avoid the transformation 
    # if not displaying the monitoring - first() starts the lazy eval)
    n_total = (df.agg(F.sum("count")).first()[0]
               if check_binary_flag_widget(display)
               else 0)
    df_ratio = (df
                .withColumn("ratio", F.round(F.col("count") / F.lit(n_total), RATIO_PRECISION))
               )
    
    return df_ratio

# COMMAND ----------

@dp.transformation(hits_per_single_keyword_ratio, display=False)
@display_result(display=DISPLAY_MONITORING_SINGLE)
def display_hits_per_single_keyword(df: DataFrame, logger: Logger):
    n_hits = df.agg(F.sum("count")).first()[0]
    logger.info(f"Total number of hits: {n_hits:,}")
    
    return df


@dp.notebook_function(hits_per_single_keyword, dp.get_widget_value("display_monitoring_single"))
def get_sum_hits_single(df, display, logger: Logger):
    if check_binary_flag_widget(display):
        n_hits = df.agg(F.sum("count")).first()[0]
        logger.info(f"Total number of single-keyword hits: {n_hits:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Bigrams
# MAGIC 
# MAGIC Displaying number of hits of each bigram:

# COMMAND ----------

@dp.transformation(load_url_level, get_keywords_bigrams_only, display=False)
def prepare_url_bigram_hits(df: DataFrame, bigrams):
    # count number of hits for each bigram
    df_hits = (df 
               .withColumn("bigram_data", F.explode("BIGRAMS"))
               .withColumn("bigram_def", F.explode(F.array(*map(F.lit, bigrams))))
               # compare bigrams both as is and reversed 
               .filter((F.col("bigram_def") == F.col("bigram_data"))
                        | (udf_reverse_word_order("bigram_def") == F.col("bigram_data")))
              )
    
    return df_hits

# COMMAND ----------

@dp.transformation(prepare_url_bigram_hits, get_keywords_bigrams_only, display=False)
def hits_per_bigram(df_hits: DataFrame, bigrams, logger: Logger):
    if len(bigrams) == 0: 
        logger.warning("Skipping: there are no keywords in the interest definition.")
        return None
    
    # prepare all bigrams as a DF
    df_bigrams = spark.createDataFrame(data=[Row(x) for x in bigrams], schema=["bigram_def"])
    # group by bigrams from the definition & count number of hits
    df_hits = (df_hits
               .groupBy("bigram_def")
               .count()
               .join(df_bigrams, "bigram_def", "right")
               .fillna(0)
               .orderBy(F.desc("count"))
              )
    
    return df_hits

# COMMAND ----------

@dp.transformation(hits_per_bigram, dp.get_widget_value("display_monitoring_bigram"), display=False)
def hits_per_bigram_ratio(df: DataFrame, display):
    if df is None:
        return None
    # add percentage column (possibly dummy, avoid the transformation 
    # if not displaying the monitoring - first() starts the lazy eval)
    n_total = (df.agg(F.sum("count")).first()[0]
               if check_binary_flag_widget(display)
               else 0)
    df_ratio = (df
                .withColumn("ratio", F.round(F.col("count") / F.lit(n_total), RATIO_PRECISION))
               )
    
    return df_ratio

# COMMAND ----------

@dp.transformation(hits_per_bigram_ratio, display=False)
@display_result(display=DISPLAY_MONITORING_BIGRAM)
def display_hits_per_bigram_ratio(df: DataFrame):
    return df


@dp.notebook_function(hits_per_bigram, dp.get_widget_value("display_monitoring_bigram"))
def get_sum_hits_bigram(df, display, logger: Logger):
    if check_binary_flag_widget(display):
        n_hits = df.agg(F.sum("count")).first()[0]
        logger.info(f"Total number of bigram hits: {n_hits:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Interest hit ratio
# MAGIC 
# MAGIC Estimating number of positive affinities (affinity is always positive when at least 1 keyword/bigram is hit for given user): 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Count all users:

# COMMAND ----------

@dp.notebook_function(load_user_id_level, dp.get_widget_value("display_monitoring_single"), dp.get_widget_value("display_monitoring_bigram"))
def num_total_users(df: DataFrame, display_single, display_bigram): 
    return (df.select(F.countDistinct("USER_ID")).first()[0]
            if check_binary_flag_widget(display_single) or check_binary_flag_widget(display_bigram) 
            else 0)


_N_TOTAL_USERS = num_total_users.result


def _print_hit_ratio(n_positive: int): 
    print(f"[HIT RATIO] Number of positive affinities: {format_fraction(n_positive, _N_TOTAL_USERS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Prepare sets of users with positive affinities (1+ hits) for both single words and bigrams:

# COMMAND ----------

@dp.notebook_function(load_user_id_level, get_keywords_single_only, dp.get_widget_value("display_monitoring_single"))
def users_single_keyword_hits_set(df: DataFrame, keywords, display):    
    # edge case: empty keywords set leads to null column data type 
    if not check_binary_flag_widget(display) or len(keywords) == 0: 
        return []
    
    return (df
            .withColumn("token", F.explode("TOKENS"))
            .filter(F.array_contains(F.array(*map(F.lit, keywords)), F.col("token")))
            .agg(F.collect_set("USER_ID"))
            .first()[0]
           )
    
    
@dp.notebook_function(load_user_id_level, get_keywords_bigrams_only, dp.get_widget_value("display_monitoring_bigram"))
def users_bigram_hits_set(df: DataFrame, bigrams, display):
    # edge case: empty keywords set leads to null column data type 
    if not check_binary_flag_widget(display) or len(bigrams) == 0: 
        return []
    
    return (df
            .withColumn("bigram_data", F.explode("BIGRAMS"))
            .withColumn("bigram_def", F.explode(F.array(*map(F.lit, bigrams))))
            # compare bigrams both as is and reversed 
            .filter((F.col("bigram_def") == F.col("bigram_data"))
                     | (udf_reverse_word_order("bigram_def") == F.col("bigram_data")))
            .agg(F.collect_set("USER_ID"))
            .first()[0]
           )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Count positive / negative hits and ratios. Count for both separate hits and collective hits (single + bigrams).

# COMMAND ----------

@dp.notebook_function(users_single_keyword_hits_set, dp.get_widget_value("display_monitoring_single"))
def single_keywords_hit_ratio(users, display): 
    if not check_binary_flag_widget(display):
        return 0
    
    n_positive = len(users)
    _print_hit_ratio(n_positive)
    
    return n_positive

# COMMAND ----------

@dp.notebook_function(users_bigram_hits_set, dp.get_widget_value("display_monitoring_bigram"))
def bigrams_hit_ratio(users, display): 
    if not check_binary_flag_widget(display):
        return 0
    
    n_positive = len(users)
    _print_hit_ratio(n_positive)
    
    return n_positive

# COMMAND ----------

@dp.notebook_function(users_single_keyword_hits_set, users_bigram_hits_set, dp.get_widget_value("display_monitoring_single"), dp.get_widget_value("display_monitoring_bigram"))
def collective_hit_ratio(users_single, users_bigram, display_single, display_bigram): 
    if (not check_binary_flag_widget(display_single)) and (not check_binary_flag_widget(display_bigram)):
        return 0
    
    users = set(users_single).union(set(users_bigram))
    
    n_positive = len(users)
    _print_hit_ratio(n_positive)
    
    return n_positive

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Summary
# MAGIC 
# MAGIC How much the bigrams are expanding the positive affinity population. 
# MAGIC 
# MAGIC Note: Neither ratio is generally wrong, but most of the interests are required to target large population.

# COMMAND ----------

@dp.notebook_function(single_keywords_hit_ratio, bigrams_hit_ratio, collective_hit_ratio, num_total_users, dp.get_widget_value("display_monitoring_single"), dp.get_widget_value("display_monitoring_bigram"))
def hit_ratio_summary(n_single, n_bigrams, n_combined, n_total_users, display_single, display_bigram): 
    if (not check_binary_flag_widget(display_single)) and (not check_binary_flag_widget(display_bigram)):
        return
    
    print("Hit ratios summary\n--------------------------")
    print(f"single keywords:  {format_fraction(n_single, n_total_users)}")
    print(f"bigrams:  {format_fraction(n_bigrams, n_total_users)}")
    print(f"combined:  {format_fraction(n_combined, n_total_users)}")
    n_gain = n_combined - n_single
    n_overlap = n_bigrams - n_gain
    print(f"bigrams overlap: {format_fraction(n_overlap, n_bigrams)}")
    print(f"gain using bigrams: + {format_fraction(n_gain, n_single)}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Definition keywords JLH
# MAGIC 
# MAGIC Computing JLH score for all keywords in the interest defintion.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Join single words flags and bigram flags together (Either hit counts as it certainly leads to positive affinity) and compute JLH score using the joint flag.

# COMMAND ----------

@dp.transformation(prepare_jhl_flags_single, prepare_jhl_flags_bigram, display=False)
def prepare_jhl_flags_single_joint(df_single: DataFrame, df_bigrams: DataFrame):
    # prepare bigram flags for the join
    df_bigrams = (df_bigrams
                  .select(F.col("USER_ID"), 
                          F.col("LABEL").alias("LABEL_BIGRAMS"))
                 )
    # join them on the single word flags
    return (df_single
            .withColumnRenamed("LABEL", "LABEL_SINGLE")
            .join(df_bigrams, on="USER_ID", how="outer")
            .fillna(0)
            .withColumn("LABEL", F.when(F.col("LABEL_SINGLE") + F.col("LABEL_BIGRAMS") > 0, 1).otherwise(0))
           )

# COMMAND ----------

@dp.transformation(prepare_jhl_flags_single_joint, display=False)
def single_jlh_score_joint_flags(df: DataFrame):
    return jlh(df=df, data_col="TOKENS", doc_id_col="USER_ID")

@dp.transformation(prepare_jhl_flags_single_joint, display=False)
def bigram_jlh_score_joint_flags(df: DataFrame):
    return jlh(df=df, data_col="BIGRAMS", doc_id_col="USER_ID")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Single tokens JLH score
# MAGIC 
# MAGIC Score for keywords and separate words from bigrams that appear in the definition.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Create a DF with all (flattened) keywords & append their origin:
# MAGIC 
# MAGIC Origin possibilities:
# MAGIC * *SINGLE* - Only contained in single keywords in the definition
# MAGIC * *BIGRAM* - Only part of a bigram (or more bigrams)
# MAGIC * *MIXED* - Part of both

# COMMAND ----------

@dp.transformation(get_keywords_all_flat, get_keywords_single_only, get_keywords_bigrams_only_flat, display=False)
def build_tokens_origin_dataframe(kws_all, kws_single, kws_bigrams, logger: Logger): 
    # bool flag for token origin
    kws_all = list(kws_all)
    origin = [("MIXED" if kw in kws_bigrams else "SINGLE") 
              if kw in kws_single else "BIGRAM" 
              for kw in kws_all]
    # convert to series
    kws_all_s, origin_s = (pd.Series(kws, dtype=object) for kws in (kws_all, origin))
    # build the DF
    df = pd.DataFrame()
    df["token"] = kws_all_s
    df["origin"] = origin_s
    
    try:
        return spark.createDataFrame(df)
    except ValueError: 
        logger.warning("Result is empty - returning `None` instead.")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Display scores of filtered separate words:

# COMMAND ----------

@dp.transformation(single_jlh_score_joint_flags, build_tokens_origin_dataframe, get_keywords_all_flat, display=False)
@display_result(display=DISPLAY_MONITORING_SINGLE)
def definition_tokens_jlh_scores(df_jlh: DataFrame, df_tokens_origin: DataFrame, kws_all):  
    if df_tokens_origin is None:
        return None
    # filter to tokens from definition & join with the origin info
    df_tokens = (df_jlh
                .filter(F.col("token").isin(kws_all))
                .join(df_tokens_origin, on="token", how="left")
                .sort(F.desc("JLH_score"))
               )
    
    return df_tokens

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Words with no score available (probably because of its low frequency):

# COMMAND ----------

@dp.transformation(definition_tokens_jlh_scores, build_tokens_origin_dataframe, display=False)
@display_result(display=DISPLAY_MONITORING_SINGLE)
def definition_tokens_jlh_missing(df_jlh: DataFrame, df_tokens_origin: DataFrame):
    if df_tokens_origin is None:
        return None
    # only keep the rows which do not appear in the JLH scores DF
    kws_present = set([row.token for row in df_jlh.collect()])
    df_missing = (df_tokens_origin
                  .filter(~F.col("token").isin(kws_present))
                 )
    
    return df_missing

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Bigrams tokens JLH score
# MAGIC 
# MAGIC Score for bigrams that appear in the definition.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Display scores of filtered bigrams:

# COMMAND ----------

@dp.transformation(bigram_jlh_score_joint_flags, get_keywords_bigrams_only, display=False)
@display_result(display=DISPLAY_MONITORING_BIGRAM)
def definition_bigrams_jlh_scores(df_jlh: DataFrame, bigrams):      
    # filter to tokens from definition & join with the origin info
    df_tokens = (df_jlh
                 .withColumnRenamed("token", "bigram")
                 .filter(F.col("bigram").isin(bigrams))
                 .sort(F.desc("JLH_score"))
                )
    
    return df_tokens

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Bigrams with no score available (probably because of its low frequency):

# COMMAND ----------

@dp.transformation(definition_bigrams_jlh_scores, get_keywords_bigrams_only, display=False)
@display_result(display=DISPLAY_MONITORING_BIGRAM)
def definition_bigrams_jlh_missing(df_jlh: DataFrame, bigrams_all, logger: Logger):
    # get list of boolean flags
    bigrams_all = list(bigrams_all)
    bigrams_present = set([row.bigram for row in df_jlh.collect()])
    is_missing_flags = [(bigram not in bigrams_present) for bigram in bigrams_all] 
    # create new DF with 1 column with bigrams which are not present in the scores DF
    df_bigrams_missing = (pd.Series(list(bigrams_all), dtype="object")
                          .to_frame()
                          .loc[is_missing_flags]
                         )
    df_bigrams_missing.columns = ["bigram"]
    
    try:
        return spark.createDataFrame(df_bigrams_missing)
    except ValueError:
        logger.warning("Result is empty - returning `None` instead.")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### URL monitoring
# MAGIC 
# MAGIC Monitoring of most common URLs & their active tokens. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Preparation of active tokens per URL

# COMMAND ----------

@dp.transformation(load_url_level, '%processing.options.use_bigrams%', display=False)
def tokens_url_normalized_concat(df: DataFrame, use_bigrams: bool):
    # TODO: names will change after the pipeline refactoring - aliases can be omitted in the future
    df_concat = (df
                 .withColumn("TOKENS_ALL", (F.concat("TOKENS", "BIGRAMS")
                                            if use_bigrams
                                            else F.col("TOKENS")
                                           )
                            )
                 .drop("TOKENS", "BIGRAMS")
                )
    
    return df_concat

# COMMAND ----------

@dp.transformation(tokens_url_normalized_concat, get_keywords_all, display=False)
def active_tokens_url_normalized_concat(df: DataFrame, keywords: list):
    df_filtered = (df
                   .withColumn("TOKEN", F.explode("TOKENS_ALL"))
                   .filter(F.col("TOKEN").isin(keywords))
                  )
    
    return df_filtered

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Common URLs

# COMMAND ----------

@dp.transformation(active_tokens_url_normalized_concat, dp.get_widget_value("top_n_url"), display=False)
@display_result(display=DISPLAY_URL_MONITORING)
def most_common_url_normalized(df: DataFrame, top_n_url):
    df_result = url_hits_for_group(df, data_col="TOKEN", group_by_col="URL_NORMALIZED")
    
    return df_result.limit(int(top_n_url))

# COMMAND ----------

@dp.transformation(active_tokens_url_normalized_concat, dp.get_widget_value("top_n_url"), display=False)
@display_result(display=DISPLAY_URL_MONITORING)
def most_common_url_level_1(df: DataFrame, top_n_url):
    df_result = url_hits_for_group(df, data_col="TOKEN", group_by_col="URL_DOMAIN_1_LEVEL")
    
    return df_result.limit(int(top_n_url))

# COMMAND ----------

@dp.transformation(active_tokens_url_normalized_concat, dp.get_widget_value("top_n_url"), display=False)
@display_result(display=DISPLAY_URL_MONITORING)
def most_common_url_level_2(df: DataFrame, top_n_url):
    df_result = url_hits_for_group(df, data_col="TOKEN", group_by_col="URL_DOMAIN_2_LEVEL")
    
    return df_result.limit(int(top_n_url))
