from pyspark.sql.column import Column
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


def standardize_column_positive(col: str) -> Column:
    '''
    Standardizes a chosen pyspark column using a sigmoid function (squash into (0, 1) interval). 
    See link: https://www.desmos.com/calculator/xnuscvrzd5


    :param col: name of pyspark column to standardize
    :return: standardized pyspark column
    '''
    return (F.col(col) / (2 * (1 + F.abs(F.col(col)))) + 1 / 2)


def standardize_column_sigmoid(col: str, sharpness: float) -> Column:
    '''
    Standardizes a chosen pyspark column using a sigmoid function (squash into (-1, 1) interval).
    See link: https://www.desmos.com/calculator/nq0okdppah

    :param col: name of pyspark column to standardize
    :param sharpness: float coefficient that controls the sharpness of standardizing function
    :return: standardized pyspark columns
    '''
    return (sharpness * F.col(col) / F.sqrt(1 + sharpness**2 * F.col(col)**2))


def calculate_count_coefficient(col: Column) -> Column:
    '''
    Custom function designed empirically to give bigger URL weight to users with more sites visited.
    Constants are designed in such a way so the function rises more quickly at smaller numbers, the rise slows down as number of sites rises and maxes out at 30.

    :param col: pyspark column containing number of sites visited
    :return: pyspark column with scaling count coefficients
    '''

    return F.when(col <= 30, (col - 3 / 4)**(1 / 4)).otherwise((30 - 3 / 4)**(1 / 4))


def convert_traits_to_location_features(df: DataFrame) -> DataFrame:
    '''
    Executes logical transformation from cpex geolocation traits to a location feature.
    
    :param df: pyspark dataframe containing boolean location flags (prague, countryside, regional_town) and number of locations (num_locations)
    :return: pyspark dataframe containing categorical feature with location
    '''
    return (df
            .withColumn("location_col",
                        F.when(((F.col("num_locations") >= 1) & (~F.col("city_flag")) & ~((F.col("prague_flag")))), "countryside")
                        .when(((F.col("num_locations") == 2) & (F.col("city_flag")) & (~F.col("prague_flag"))), "regional_town")
                        .otherwise("none"))
           )
    