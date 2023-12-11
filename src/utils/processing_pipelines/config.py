"""
Config of included pipeline stages based on the version of the processing used
"""

from adpickercpex.utils.processing_pipelines._processing_strategy import ProcessingStrategy

    
# ---------------------------------------- stages config ----------------------------------------


def processing_stages_data(processing_strategy: ProcessingStrategy) -> tuple:
    """
    Static function that returns set of stages that shall be included in the 
    data processing pipeline based on the processing version.
    
    :param processing_strategy: how to process the interest definitions
    :return: names of flag paramaters of the 'run_data_processing_pipeline' function that shall be set to True
    """
    return {
        ProcessingStrategy.CLEANED_SINGLE.value: (
            "include_string_imputer",
            "include_tokenizer",
            "include_words_remover",
            "include_diacritics_stripper",
        ),
        ProcessingStrategy.STEMMED_SINGLE.value: (
            "include_string_imputer",
            "include_tokenizer",
            "include_words_remover",
            "include_diacritics_stripper",
            "include_stemmer",
        ),
        ProcessingStrategy.CLEANED_BIGRAM.value: (
            "include_string_imputer",
            "include_tokenizer",
            "include_words_remover",
            "include_diacritics_stripper",
            "include_bigram_assembler",
        ),
        ProcessingStrategy.STEMMED_BIGRAM.value: (
            "include_string_imputer",
            "include_tokenizer",
            "include_words_remover",
            "include_diacritics_stripper",
            "include_stemmer",
            "include_bigram_assembler",
        ),
    }.get(processing_strategy.value)
    
    
def processing_stages_interest_tokens(processing_strategy: ProcessingStrategy) -> tuple:
    """
    Static function that returns set of stages that shall be included in the 
    interest (single-word) processing pipeline.
    
    :param processing_strategy: how to process the interest definitions
    :return: names of flag paramaters of the 'run_data_processing_pipeline' function that shall be set to True
    """
    return {
        ProcessingStrategy.CLEANED_SINGLE.value: (
            "include_words_remover",
            "include_diacritics_stripper",
        ),
        ProcessingStrategy.STEMMED_SINGLE.value: (
            "include_words_remover",
            "include_diacritics_stripper",
            "include_stemmer",
        ),
        ProcessingStrategy.CLEANED_BIGRAM.value: (
            "include_words_remover",
            "include_diacritics_stripper",
        ),
        ProcessingStrategy.STEMMED_BIGRAM.value: (
            "include_words_remover",
            "include_diacritics_stripper",
            "include_stemmer",
        ),
    }.get(processing_strategy.value)
    
    
def processing_stages_interest_bigrams(processing_strategy: ProcessingStrategy) -> tuple:
    """
    Static function that returns set of stages that shall be included in the 
    interest (bigram) processing pipeline.
    
    :param processing_strategy: how to process the interest definitions
    :return: names of flag paramaters of the 'run_bigram_processing_pipeline' function that shall be set to True
    """
    return {
        ProcessingStrategy.CLEANED_SINGLE.value: (),
        ProcessingStrategy.STEMMED_SINGLE.value: (),
        ProcessingStrategy.CLEANED_BIGRAM.value: (
            "include_words_remover",
            "include_diacritics_stripper",
        ),
        ProcessingStrategy.STEMMED_BIGRAM.value: (
            "include_words_remover",
            "include_diacritics_stripper",
            "include_stemmer",
        ),
    }.get(processing_strategy.value)
    
    
# ---------------------------------------- default column names ----------------------------------------


col_factory_imputed = lambda x: f"{x}_IMPUTED"
col_factory_tokenized = lambda x: f"{x}_TOKENIZED"
col_factory_removed = lambda x: f"{x}_STOP_REMOVED"
col_factory_stripped = lambda x: f"{x}_STRIPPED"
col_factory_stemmed = lambda x: f"{x}_STEMMED"
col_factory_bigrams = lambda x: f"{x}_BIGRAMS"



