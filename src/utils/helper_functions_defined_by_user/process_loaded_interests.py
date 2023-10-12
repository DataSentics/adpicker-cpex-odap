"""

TODO: 
This file is obsolete - it is not deleted just so that the main pipeline still works.
It should be deleted after the main pipeline is refactored.
"""

import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame
from collections import namedtuple

from  src.utils.helper_functions_defined_by_user._functions_nlp import df_stemming
from  src.utils.helper_functions_defined_by_user._stop_words import unwanted_tokens

chars_from = (
    "ÀÁÂÃÄÅČÇĎÈÉÊËÍÌÏÎĹĽŇÑÒÓÔÕÖŔŘŠŤÙÚŮÛŰÜÝŸŽàáâãäåčçďèéêëíìïîĺľňñòóôõöŕřšťùúůûűüýÿž"
)
chars_to = (
    "AAAAAACCDEEEEIIIILLNNOOOOORRSTUUUUUUYYZaaaaaaccdeeeeiiiillnnooooorrstuuuuuuyyz"
)

Interest = namedtuple("Interest", ["keywords", "general_interest"])

def _perform_cleaning_stemming(df: DataFrame, keyword_col: str):
    df = df.withColumn(keyword_col, F.translate(keyword_col, chars_from, chars_to))

    return df_stemming(
        df=df,
        input_col=keyword_col,
        cleaned_col='keyword_cleaned',
        stemmed_col='keyword_stemmed',
        client_name='cz',
        stop_words=unwanted_tokens
    )


def _cleaned_stemmed_keywords(df: DataFrame, input_col: str, output_col: str, keyword_variant: str, bigrams=False):
    # perform cleaning and stemming of keywords
    df = df.withColumn('keyword', F.explode(input_col)).drop('keywords')

    df = _perform_cleaning_stemming(df=df, keyword_col='keyword')

    if bigrams:
        # put bigrams back together
        df = (df
              .withColumn('keyword_cleaned', F.array(F.concat_ws(" ", F.col('keyword_cleaned'))))
              .withColumn('keyword_stemmed', F.array(F.concat_ws(" ", F.col('keyword_stemmed'))))
             )

    df = (df
      .groupBy('subinterest', 'general_interest')
      .agg(
          F.array_distinct(F.flatten(F.collect_list('keyword_cleaned'))).alias('keywords_cleaned'),
          F.array_distinct(F.flatten(F.collect_list('keyword_stemmed'))).alias('keywords_stemmed'),
      )
     )

    if keyword_variant != 'stemmed':
        keywords_col = 'keywords_cleaned'
    else:
        keywords_col = 'keywords_stemmed'

    return (df
            .select(
                'subinterest',
                'general_interest',
                F.col(keywords_col).alias(output_col)
            )
           )


def process_loaded_interests(df: DataFrame, general_interests=False, use_bigrams=False, keyword_variant='cleaned', name_prefix='ad_interest_affinity_'):
    # prepeare mapping between feature store and db
    mapping_to_db = {
        name_prefix + row['subinterest'].lower(): str(row['db_id'])
        for row in df.collect()
    }

    # perform cleaning/stemming
    df_kws = _cleaned_stemmed_keywords(df=df, input_col='keywords', output_col='keywords', keyword_variant=keyword_variant, bigrams=False)

    if general_interests:
        df_kws = (df_kws
                  .groupBy('general_interest')
                  .agg(
                      F.array_distinct(F.flatten(F.collect_list('keywords'))).alias('keywords'),
                  )
                  .withColumn('subinterest', F.col('general_interest'))
                 )

    if use_bigrams:
        df_kws_bi = _cleaned_stemmed_keywords(df=df, input_col='keywords_bigrams', output_col='keywords_bigrams', keyword_variant=keyword_variant, bigrams=True)

        if general_interests:
            df_kws_bi = (df_kws_bi
                         .groupBy('general_interest')
                         .agg(
                             F.array_distinct(F.flatten(F.collect_list('keywords_bigrams'))).alias('keywords_bigrams'),
                         )
                         .withColumn('subinterest', F.col('general_interest'))
                        )

        df_kws = (df_kws
                  .join(df_kws_bi, on=['subinterest', 'general_interest'])
                  .withColumn('keywords', F.concat(F.col('keywords'), F.col('keywords_bigrams')))
                  .drop('keywords_bigrams')
                 )

    df_pd = df_kws.toPandas()

    # prepare interests' named tuple
    interests_tuple = {
        name_prefix + row['subinterest'].lower(): Interest(
            row['keywords'].tolist(), row['general_interest']
        )
        for _, row in df_pd.iterrows()
    }

    # interest names
    interests_names = list(interests_tuple.keys())

    return {
        'tuple': interests_tuple,
        'names': interests_names,
        'mapping_to_db': mapping_to_db,
    }

