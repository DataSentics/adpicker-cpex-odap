"""
Definition of processing pipelines and wrapper functions that allow to build and run 
the pre-defined pipelines with an option to skip some stages
"""

from logging import Logger
from dataclasses import dataclass, field

from pyspark.sql.dataframe import DataFrame
from pyspark.ml.feature import NGram

from pyspark.ml import Transformer, Pipeline 
from pyspark.ml.feature import StopWordsRemover, RegexTokenizer

from src.utils.helper_functions_defined_by_user._stop_words import unwanted_tokens

from src.utils.processing_pipelines._transformers import (
    StringImputer,
    DiacriticsStripper, 
    WordStemmer, 
    NGramStopwordsRemover, 
    NGramDiacriticsStripper, 
    NGramStemmer,
)

from src.utils.processing_pipelines import config

# ---------------------------------------- Processing pipeline builders ----------------------------------------

@dataclass(frozen=True)
class PipelineStageInfo:
    """
    Stores info about one transformer stage of a pyspark pipeline
    """
    
    """
    transformer class to be used
    """
    transformer: Transformer
    """
    name of the output column of the stage
    """
    output_col: str
    """
    stage tag (for logging)
    """
    tag: str = field(default_factory=lambda: "<UNDEFINED_TAG>")
    """
    Additional keyword arguments of the transformer (apart from input & output columns)
    """
    transformer_kwargs: dict = field(default_factory=lambda: {})
    

def _build_pipeline(input_col: str, 
                    *args: PipelineStageInfo, 
                    logger: Logger = None) -> Pipeline:
    """
    Builds a pipeline from `PipelineStageInfo` objects passed as positional arguments. The ordering is maintained.
    
    :param input_col: name of the input column of the pipeline
    :param args: each object represents info needed to initialize one stage of the pipeline 
    :param logger: logger instance
    :return: spark pipeline with the stages defined by the `args` 
    """
    # use outputs of previous transformers as inputs for the following ones
    def _get_input_column():
        return input_col if len(stages) == 0 else stages[-1].getOutputCol()
    
    # iteratively initialize stages
    stages = []
    for info in args:
        if info.output_col is not None:
            stages.append(info.transformer(inputCol=_get_input_column(),
                                           outputCol=info.output_col,
                                           **info.transformer_kwargs))
    if logger is not None: 
        tags = [info.tag for info in args if info.output_col is not None]
        n_stages = len(tags)
        if n_stages != 0:
            logger.info(f"Pipeline for `{input_col}` with {n_stages} stages built: {' -> '.join(tags)}")
    
    return Pipeline(stages=stages)
    

def build_data_processing_pipeline(input_col: str, 
                                   string_imputer_col: str = None, 
                                   tokenizer_output_col: str = None, 
                                   stopwords_remover_output_col: str = None, 
                                   diacritics_stripper_output_col: str = None, 
                                   stemmer_output_col: str = None, 
                                   bigrams_output_col: str = None,
                                   logger: Logger = None) -> Pipeline: 
    """
    Returns a spark pipeline that includes stages according to the passed column names.
    When there is no column name passed for a stage of the pipeline, the stage is skipped.
    
    The full pipeline goes as follows (stage 1 to 6): 
    <nulls imputation> -> <tokenizing> -> <unwanted tokens removing> -> <diacritics stripping> -> <stemming> -> <assemble of bigrams>
    
    This pipeline is assumed to be used on the data. The cleaning/stemming part can be used on single-word keywords from interest definitions.
   
    :param input_col: input column of the pipeline
    :param string_imputer_col: name of the new column with nulls imputed by empty strings
    :param tokenizer_output_col: name of the new column with raw tokens
    :param stopwords_remover_output_col: name of the new column with tokens that do not contains stop-words
    :param diacritics_stripper_output_col: name of the new column with diacritics-stipped (cleaned) tokens
    :param stemmer_output_col: name of the new column with stemmed tokens
    :param bigrams_output_col: name of the new column with assembled bigrams of the pipeline
    :return: spark Pipeline with all defined steps
    """
    stages_included = [
        PipelineStageInfo(transformer=StringImputer,
                          output_col=string_imputer_col,
                          tag="NULL_STRING_IMPUTER",
                         ),
        PipelineStageInfo(transformer=RegexTokenizer,
                          output_col=tokenizer_output_col,
                          tag="STRING_TOKENIZER",
                          transformer_kwargs={
                              "pattern": r"[\W_]+",
                              "minTokenLength": 2,
                          }
                         ),
        PipelineStageInfo(transformer=StopWordsRemover,
                          output_col=stopwords_remover_output_col,
                          tag="UNWANTED_WORDS_REMOVER",
                          transformer_kwargs={
                              "stopWords": unwanted_tokens,
                          }
                         ),
        PipelineStageInfo(transformer=DiacriticsStripper,
                          output_col=diacritics_stripper_output_col,
                          tag="DIACRITICS_STRIPPER",
                         ),
        PipelineStageInfo(transformer=WordStemmer,
                          output_col=stemmer_output_col,
                          tag="WORD_STEMMER",
                         ),
        PipelineStageInfo(transformer=NGram,
                          output_col=bigrams_output_col,
                          tag="BIGRAM_ASSEMBLER",
                          transformer_kwargs={
                              "n": 2,
                          }
                         ),
    ]
    
    pipeline = _build_pipeline(input_col,
                               *stages_included, 
                               logger=logger)
    
    return pipeline


def build_bigram_processing_pipeline(input_col: str, 
                                     stopwords_remover_output_col: str = None, 
                                     diacritics_stripper_output_col: str = None, 
                                     stemmer_output_col: str = None,
                                     logger: Logger = None) -> Pipeline: 
    """
    Returns a spark pipeline that includes stages according to the passed column names.
    When there is no column name passed for a stage of the pipeline, the stage is skipped
    
    The full pipeline goes as follows (stage 1 to 3): 
    <unwanted tokens removing> -> <diacritics stripping> -> <stemming>
   
    This pipeline is assumed to be used solely on the bigram keywords from interest definitions. 
    
    :param input_col: input column of the pipeline
    :param stopwords_remover_output_col: name of the new column with bigrams that do not contains stop-words (they might become monograms/empty)
    :param diacritics_stripper_output_col: name of the new column with diacritics-stipped (cleaned) bigrams
    :param stemmer_output_col: name of the new column with stemmed bigrams
    :return: spark Pipeline with all defined steps
    """
    stages_included = [
        PipelineStageInfo(transformer=NGramStopwordsRemover,
                          output_col=stopwords_remover_output_col,
                          tag="NGRAM_UNWANTED_WORDS_REMOVER",
                          transformer_kwargs={
                              "stopWords": unwanted_tokens,
                          }
                         ),
        PipelineStageInfo(transformer=NGramDiacriticsStripper,
                          output_col=diacritics_stripper_output_col,
                          tag="NGRAM_DIACRITICS_STRIPPER",
                         ),
        PipelineStageInfo(transformer=NGramStemmer,
                          output_col=stemmer_output_col,
                          tag="NGRAM_WORD_STEMMER",
                         ),
    ]
    
    pipeline = _build_pipeline(input_col,
                               *stages_included, 
                               logger=logger)
    
    return pipeline

# ---------------------------------------- processing pipeline runners ----------------------------------------


def _apply_pipeline(df: DataFrame, 
                    pipeline: Pipeline,
                    logger: Logger = None) -> DataFrame:
    """
    Wrapper that applies the pipeline to the DF
    
    :param df: spark DF
    :param pipeline: pipeline to be applied (fit & transformed)
    :param logger: logger instance
    :return: transformed DF
    """
    # skip when there are no stages
    n_stages = len(pipeline.getStages())
    if n_stages == 0: 
        return df
    # run the pipeline & log changes
    n_cols_old = len(df.columns)
    df_new = pipeline.fit(df).transform(df)
    n_cols_new = len(df_new.columns)
    if logger is not None:
        logger.debug(f"Pipeline of {n_stages} stages ran successfully. "
                     f"Change in number of columns: {n_cols_old} -- ({n_cols_new - n_cols_old:+}) --> {n_cols_new}")
    
    return df_new


def run_data_processing_pipeline(df: DataFrame,
                                 input_col: str,
                                 include_string_imputer: bool = False,
                                 include_tokenizer: bool = False,
                                 include_words_remover: bool = False,
                                 include_diacritics_stripper: bool = False,
                                 include_stemmer: bool = False,
                                 include_bigram_assembler: bool = False,
                                 logger: Logger = None) -> DataFrame:
    """
    Runs the pre-defined data processing pipeline with selected stages (the order of the stages cannot be changed).
    Uses the `build_data_processing_pipeline` to build the pipeline.
    
    The stages of the data processing pipeline: 
    <nulls imputation> -> <tokenizing> -> <unwanted tokens removing> -> <diacritics stripping> -> <stemming> -> <assemble of bigrams>
    
    :param df: spark DF 
    :param input_col: input column of the pipeline
    :param include_string_imputer: stage selection flag - impute nulls with empty strings
    :param include_tokenizer: stage selection flag - create tokens from a text
    :param include_words_remover: stage selection flag - remove unwanted tokens from a set 
    :param include_diacritics_stripper: stage selection flag - strip all tokens of diacritics
    :param include_stemmer: stage selection flag - apply stemming to all tokens
    :param include_bigram_assembler: stage selection flag - assemble all possible bigrams
    :param logger: logger instance 
    :return: 
    """
    # create a kwargs dictionary for the pipeline builder (format: {param: value, flag})
    stages_map = {
        "string_imputer_col": (config.col_factory_imputed(input_col), include_string_imputer),
        "tokenizer_output_col": (config.col_factory_tokenized(input_col), include_tokenizer),
        "stopwords_remover_output_col": (config.col_factory_removed(input_col), include_words_remover),
        "diacritics_stripper_output_col": (config.col_factory_stripped(input_col), include_diacritics_stripper),
        "stemmer_output_col": (config.col_factory_stemmed(input_col), include_stemmer),
        "bigrams_output_col": (config.col_factory_bigrams(input_col), include_bigram_assembler),
    }
    # filter based on the flags (parameters)
    stages_kwargs = {param_name: column_name 
                     for param_name, (column_name, use_flag) in stages_map.items() 
                     if use_flag}
    # build the pipeline with default output column names
    pipeline = build_data_processing_pipeline(input_col=input_col, 
                                              logger=logger, 
                                              **stages_kwargs)
    # run the pipeline with the DF
    df_transformed = _apply_pipeline(df, pipeline, logger)
    
    return df_transformed


def run_bigram_processing_pipeline(df: DataFrame,
                                   input_col: str,
                                   include_words_remover: bool = False,
                                   include_diacritics_stripper: bool = False,
                                   include_stemmer: bool = False,
                                   logger: Logger = None) -> DataFrame:
    """
    Runs the pre-defined bigram processing pipeline with selected stages (the order of the stages cannot be changed).
    Uses the `build_bigram_processing_pipeline` to build the pipeline.
    
    The full pipeline goes as follows (stage 1 to 3): 
    <unwanted tokens removing> -> <diacritics stripping> -> <stemming>
    
    :param df: spark DF 
    :param input_col: input column of the pipeline
    :param include_words_remover: stage selection flag - remove unwanted words from separate bigrams (can become monograms/empty)  
    :param include_diacritics_stripper: stage selection flag - strip all bigrams of diacritics
    :param include_stemmer: stage selection flag - apply stemming to separate words in the bigrams
    :param logger: logger instance 
    """
    # create a kwargs dictionary for the pipeline builder (format: {param: value, flag})
    stages_map = {
        "stopwords_remover_output_col": (config.col_factory_removed(input_col), include_words_remover),
        "diacritics_stripper_output_col": (config.col_factory_stripped(input_col), include_diacritics_stripper),
        "stemmer_output_col": (config.col_factory_stemmed(input_col), include_stemmer),
    }
    # filter based on the flags (parameters)
    stages_kwargs = {param_name: column_name 
                     for param_name, (column_name, use_flag) in stages_map.items() 
                     if use_flag}
    # build the pipeline with default output column names
    pipeline = build_bigram_processing_pipeline(input_col=input_col, 
                                                logger=logger, 
                                                **stages_kwargs)
    # run the pipeline with the DF
    df_transformed = _apply_pipeline(df, pipeline, logger)
    
    return df_transformed

