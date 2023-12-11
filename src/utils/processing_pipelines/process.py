"""
Definition of processing version & high-level processing functions that 
"""

from logging import Logger
from collections.abc import Iterable

from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F

from adpickercpex.utils.processing_pipelines import config
from adpickercpex.utils.processing_pipelines.pipeline import (
    run_data_processing_pipeline,
    run_bigram_processing_pipeline,
)
from adpickercpex.utils.processing_pipelines._processing_strategy import (
    ProcessingStrategy,
    FLAG_STEMMING,
    FLAG_BIGRAM,
)


# ---------------------------------------- processing version API ----------------------------------------


def create_processing_strategy(*, 
                               use_stemming: bool, 
                               use_bigrams: bool, 
                               logger: Logger = None) -> ProcessingStrategy:
    """
    Returns a value that determines how to process data
    
    :param use_stemming: True - data are cleaned and then stemmed, False - data are only cleaned
    :param use_bigrams: True - bigrams are assembled from the data, False - bigrams are ignored 
    :return: version of data & interest processing 
    """
    flags_sum = 0b00
    if use_stemming: 
        flags_sum |= FLAG_STEMMING
    if use_bigrams: 
        flags_sum |= FLAG_BIGRAM
    # convert to enum    
    version = ProcessingStrategy(flags_sum)
    
    if logger is not None:
        logger.info(f"processing version: {version.name}")
        
    return version


def get_active_flag_stemming(processing_strategy: ProcessingStrategy) -> bool:
    """
    Getter of whether the processing version uses stemming or not
    
    :param processing_strategy: url data processing version 
    :return: boolean flag
    """
    return bool(int(processing_strategy.value) & FLAG_STEMMING)
    

def get_active_flag_bigrams(processing_strategy: ProcessingStrategy) -> bool:
    """
    Getter of whether the processing version uses bigrams or not
    
    :param processing_strategy: url data processing version 
    :return: boolean flag
    """
    return bool(int(processing_strategy.value) & FLAG_BIGRAM)


# ---------------------------------------- high-level processing functions ----------------------------------------


def _extract_col_names(kwargs: dict, col_param_names: list) -> list:
    """
    Extracts names of columns passed as keyword arguments to a function. 
    The column names can be passed either as strings or any iterables.
    
    :param kwargs: kwargs of the function
    :param col_param_names: names of parameters that contain the column names
    :return: extracted column names
    """
    col_names = []
    # iterate variables passed to the decorator, add them 
    for param_name in col_param_names:
        try:
            param_value = kwargs[param_name]
        except KeyError:
            # skip parameters that were not passed
            continue
        # append string parameters (they contain column names)
        if isinstance(param_value, str):
            col_names.append(param_value)
        # concatenate iterable parameters (they contain list-convertable column of names)
        elif isinstance(param_value, Iterable):
            col_names += list(param_value)
        else:
            raise ValueError(f"Parameter `{param_name}` that should store column name(s) contains"
                             f"neither a string nor an iterable. Type: ({type(param_value)})")
    
    return col_names
    
    
def _add_overwrite_option(input_col_param_names: list):
    """
    Decorator, adds `overwrite_old_names` option as an keyword argument to the decorated function
    to drop all columns created during the transformation, modifying the input columns instead.

    Expects a transformation function with a pyspark DF as the first positional argument.

    :param input_col_variable_names: names of variables that indicate input columns of the transformation
    :return: processing function decorator 
    """
    def _add_overwrite_option_inner(f):
        def _transform_overwrite(df: DataFrame, 
                                 processing_strategy: ProcessingStrategy,
                                 *, 
                                 overwrite_old_names: bool = True,
                                 **kwargs):
            """
            :param df: spark DF
            :processing_strategy: processing version to be used in the decorated transformation
            :param overwrite_old_names: keyword argument option added to the decorated function that
                                        allows modifying input columns instead of adding new ones
            :param *args: other keyword arguments of the decorated function
            :return: decorated function
            """
            # store original columns
            cols_original = df.columns
            # transform
            df_transformed = f(df, processing_strategy, **kwargs)

            if not overwrite_old_names:
                # keep temporary columns
                return df_transformed
            
            # find output column name based on the processing version
            output_col_factory = (config.col_factory_stemmed 
                                  if get_active_flag_stemming(processing_strategy) 
                                  else config.col_factory_stripped)
            # get names of the input columns, rename (overwrite) them all if they exist: output_col -> input_col
            for input_col in _extract_col_names(kwargs, input_col_param_names):
                output_col = output_col_factory(input_col)
                if output_col not in df_transformed.columns:
                    continue
                df_transformed = df_transformed.withColumn(input_col, F.col(output_col))
            # drop temprary columns created during the transformation
            df_cols_original = df_transformed.select(*cols_original)

            if "logger" in kwargs:
                logger = kwargs["logger"]
                if logger is not None and isinstance(logger, Logger):
                    logger.debug("Temporary columns created during the processing dropped - "
                         f"final number of columns: {len(df_cols_original.columns)}")

            return df_cols_original

        return _transform_overwrite
    
    return _add_overwrite_option_inner
    
    
@_add_overwrite_option(input_col_param_names=["input_cols"])
def process_data(df: DataFrame, 
                 processing_strategy: ProcessingStrategy,
                 *,
                 input_cols: list, 
                 logger: Logger = None) -> DataFrame:
    """
    Processes each column separetly according to the processing version, creating new column(s)
    The pipeline that is used to transform the DF depends on the processing version.
    
    :param df: pyspark DF 
    :param processing_strategy: how to process the data
    :param input_cols: name of the columns that will be processed
    :param logger: logger instance 
    :return: pyspark DF with processed columns
    """
    # get names of stages that should be included in the pipeline
    stage_flags = {param_name: True 
                   for param_name in config.processing_stages_data(processing_strategy)}
    # run the pipeline separately for all columns
    for input_col in input_cols:
        df = run_data_processing_pipeline(df, 
                                          input_col=input_col, 
                                          logger=logger, 
                                          **stage_flags)
    return df
    

@_add_overwrite_option(input_col_param_names=["input_col_single", "input_col_bigrams"])
def process_interest_definitions(df: DataFrame, 
                                 processing_strategy: ProcessingStrategy,
                                 *,
                                 input_col_single: str,
                                 input_col_bigrams: str, 
                                 index_col: str = "subinterest",
                                 logger: Logger = None) -> DataFrame:
    """
    Processes both single words and bigrams of interests definitions so that they can be matched with the data.
    The pipeline that is used to transform the DF depends on the processing version.
    
    Before the transformation, the index column is set to lower case to be surely consistent across the whole project. 
    
    :param df: loaded pyspark DF with interest definitions
    :param processing_strategy: how to process the interest definitions
    :param single_col: name of the column that contains single words
    :param bigrams_col: name of the column that contains bigrams
    :param index_col: name of the column that identifies unique interest
    :param overwrite_old_names: [decorator keyword argument] if True, no columns are created, the input columns are modified instead
    :param logger: logger instance 
    :return: pyspark DF with transformed keywords
    """
    # transform single-word keywords
    flags_single = {param_name: True 
                    for param_name in config.processing_stages_interest_tokens(processing_strategy)}
    df_single = run_data_processing_pipeline(df,
                                             input_col=input_col_single,
                                             logger=logger,
                                             **flags_single)
    # transform bigrams
    flags_double = {param_name: True
                    for param_name in config.processing_stages_interest_bigrams(processing_strategy)}
    df_bigrams = run_bigram_processing_pipeline(df_single,
                                                input_col=input_col_bigrams,
                                                logger=logger,
                                                **flags_double)
    # set index to lower-case
    df_idx_lower = df_bigrams.withColumn(index_col, F.lower(index_col))
            
    return df_idx_lower


    