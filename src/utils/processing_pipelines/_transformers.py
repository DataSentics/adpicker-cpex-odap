"""
Definitions of custom transformers that are used in processing pipelines 

The definitions of all custom transformer classes are simplified. Example of more complex definition: 
https://stackoverflow.com/questions/32331848/create-a-custom-transformer-in-pyspark-ml
"""

from abc import abstractmethod
    
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F

from pyspark import keyword_only
from pyspark.ml import Transformer 
from pyspark.ml.feature import StopWordsRemover
    
import adpickercpex.solutions._functions_nlp as nlp
import adpickercpex.solutions._functions_udf as adpudf
    

# ---------------------------------------- helper functions definitions ----------------------------------------


def _rename_input_if_ambiguous(df: DataFrame, 
                               input_col: str,
                               output_col: str): 
    """
    Renames the input column if it matches the output column to avoid ambiguity
    
    :param df: Spark DF
    :param input_col: input column of a pipeline
    :param output_col: output column of a pipeline
    :return: tupple of DF, input_column (both having the input column possibly renamed) 
    """
    if input_col != output_col:
        return df, input_col
    
    # rename the DF column and change the column name variable
    new_input_col = f"{input_col}_original_"
    df = df.withColumnRenamed(input_col, new_input_col)
    
    return df, new_input_col


def _transform_ngram_column(df: DataFrame, 
                            transformer_class: Transformer, 
                            input_col: str, 
                            output_col: str, 
                            **transformer_kwargs):
    """
    Uses a transformer that expects an array column of single-word keywords 
    to transform an array column of ngrams (words separated by a whitespace)
    
    :param df: spark DF
    :param transformer: pyspark transformer to be used for the transformation
    :param input_col: name of the `inputColumn` of the transformer 
    :param output_col: name of the `outputColumn` of the transformer
    :return: transformed spark DF 
    """
    # explode the original array column to have 1 bigram per row, split the bigram into separate words 
    col_exploded, col_transformed = f"{input_col}_exploded", f"{input_col}_transformed"
    df_exploded = (df
                   .withColumn(col_exploded, F.explode(input_col))
                   .withColumn(col_exploded, F.split(col_exploded, pattern=r'[\W_]+'))
                  )
    # perform the transformation on separate words
    transformer = transformer_class(inputCol=col_exploded, outputCol=col_transformed, **transformer_kwargs)
    df_transformed = transformer.transform(df_exploded)
    # concatenate the words back into n-grams (avoid naming ambiguity)
    df, input_col = _rename_input_if_ambiguous(df, input_col, output_col)
    df_concat = (df_transformed
                 .withColumn(col_transformed, F.concat_ws(" ", col_transformed))
                 .groupBy(df.columns)
                 .agg(F.collect_set(col_transformed).alias(output_col))
                )
    # join stemmed n-grams onto the original DF (necesarry to include interests with no n-grams)
    df_join = (df
               .join(df_concat, on=df.columns, how="left")
               .withColumn(output_col, F.coalesce(output_col, F.array()))
              )

    return df_join


# ---------------------------------------- transformer templates ----------------------------------------


class _BaseTransformer(Transformer): 
    """
    General one-column transformer with input/output column getters & null imputation
    """
    def __init__(self, inputCol: str, outputCol: str, imputeCol: str = None):
        """
        :param inputCol: name of the input column of the transformation
        :param outputCol: name of the output column of the transformation
        :param imputeCol: name of the column with imputed values 
                         (when left as None, the imputed column is not appended to the result)
        """
        super().__init__()
        self.inputCol = inputCol 
        self.outputCol = outputCol
        self.imputeCol = imputeCol
        
    def getInputCol(self) -> str:
        return self.inputCol
    
    def getOutputCol(self) -> str:
        return self.outputCol
    
    def getImputeCol(self) -> str:
        return self.imputeCol
        
    def _transform(self, dataset: DataFrame):
        """
        General transformation wrapper method that imputes null values before the transformation.
        It calls the abstract `self._do_transform` method that should be overwritten by all children.
        
        :param dataset: spark DF
        :return: DF with new transformed column (and possibly with the imputed column)
        """
        # impute nulls to avoid errors 
        col_original = self.inputCol
        self.inputCol = f"{self.inputCol}_imputed_tmp_" if self.imputeCol is None else self.imputeCol
        dataset = dataset.withColumn(self.inputCol, F.coalesce(col_original, self._get_impute_value()))
        
        # do the transformation
        df_transformed = self._do_transform(dataset)
        if self.imputeCol is None: 
            # the imputed column is temporary (not defined as param), remove it
            df_transformed = df_transformed.drop(self.inputCol)
            
        self.inputCol = col_original
       
        return df_transformed
    
    @abstractmethod
    def _do_transform(self, df: DataFrame):
        """
        Does the actual transformation of the DF 
        """
        
    @abstractmethod
    def _get_impute_value(self):
        """
        Returns a pyspark column that is used to impute nulls 
        """
        
    
class _StringTransformer(_BaseTransformer):
    """
    General (abstract) transformer that only transforms a single string-column of a DF, 
    appending the new transformed column to the DF at the output.  
    
    The values that would not appear in the transformed DF are imputed with empty strings instead.
    """
    def __init__(self, inputCol: str, outputCol: str, imputeCol: str = None):
        super().__init__(inputCol, outputCol, imputeCol)
    
    @abstractmethod
    def _do_transform(self, df: DataFrame):
        """
        Does the actual transformation of the DF 
        """
                 
    def _get_impute_value(self):
        return F.lit("") 

    
class _ArrayTransformer(_BaseTransformer):
    """
    General (abstract) transformer that only transforms a single array-column of a DF, 
    appending the new transformed column to the DF at the output.  
    
    The values that would not appear in the transformed DF are imputed with empty arrays instead.
    """
    def __init__(self, inputCol: str, outputCol: str, imputeCol: str = None):
        super().__init__(inputCol, outputCol, imputeCol)
    
    @abstractmethod
    def _do_transform(self, df: DataFrame):
        """
        Does the actual transformation of the DF 
        """
                 
    def _get_impute_value(self):
        return F.array() 
    
    
# ---------------------------------------- simple transformers ----------------------------------------

    
class StringImputer(_StringTransformer):
    """
    Custom transformer that imputes all nulls in a string column by an empty string
    """
    @keyword_only
    def __init__(self, inputCol: str, outputCol: str):
        # use the built-in imputation of the _BaseTransformer
        super().__init__(inputCol, outputCol, imputeCol=outputCol)
        
    def _do_transform(self, df: DataFrame) -> DataFrame:   
        # imputation is done natively
        return df
    
    
class DiacriticsStripper(_ArrayTransformer):
    """
    Custom transformer that removes diacritics (accents) from all strings in an array-column of tokens
    """
    @keyword_only
    def __init__(self, inputCol: str, outputCol: str):
        super().__init__(inputCol, outputCol)
    
    def _do_transform(self, df: DataFrame) -> DataFrame:    
        return df.withColumn(self.outputCol, adpudf.udf_strip_diacritics_array(self.inputCol))
        

class WordStemmer(_ArrayTransformer):
    """
    Custom transformer that performs stemming on an array-column of tokens
    """
    @keyword_only
    def __init__(self, inputCol: str, outputCol: str):
        super().__init__(inputCol, outputCol)
    
    def _do_transform(self, df: DataFrame) -> DataFrame:
        return df.withColumn(self.outputCol, nlp.udfstemed_lst(self.inputCol, F.lit("cz"))) 
        
        
# ---------------------------------------- bigram processing transformers ----------------------------------------


class NGramStopwordsRemover(_ArrayTransformer):
    """
    Custom transformer that removes stopwords from an array-column of n-grams
    """
    @keyword_only
    def __init__(self, inputCol: str, outputCol: str, stopWords: list):
        super().__init__(inputCol, outputCol)
        self.stopWords = stopWords
    
    def _do_transform(self, df: DataFrame) -> DataFrame:
        return _transform_ngram_column(df,
                                       transformer_class=StopWordsRemover,
                                       input_col=self.inputCol,
                                       output_col=self.outputCol,
                                       stopWords=self.stopWords)
        
        
class NGramDiacriticsStripper(_ArrayTransformer):
    """
    Custom transformer that removes diacritics (accents) from all strings in an array-column of n-grams
    """
    @keyword_only
    def __init__(self, inputCol: str, outputCol: str):
        super().__init__(inputCol, outputCol)
        
    def _do_transform(self, df: DataFrame) -> DataFrame:
        return _transform_ngram_column(df,
                                       transformer_class=DiacriticsStripper,
                                       input_col=self.inputCol,
                                       output_col=self.outputCol)
        
        
class NGramStemmer(_ArrayTransformer):
    """
    Custom transformer that performs stemming on an array-column of n-grams
    """
    @keyword_only
    def __init__(self, inputCol: str, outputCol: str):
        super().__init__(inputCol, outputCol)
    
    def _do_transform(self, df: DataFrame) -> DataFrame:
        return _transform_ngram_column(df,
                                       transformer_class=WordStemmer,
                                       input_col=self.inputCol,
                                       output_col=self.outputCol)

