import re
import pandas as pd
import unicodedata
from unidecode import unidecode
import urllib.parse as urlParse

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import udf
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.session import SparkSession
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.dbutils import DBUtils

from pyspark.ml.feature import StopWordsRemover, RegexTokenizer
from pyspark.ml import Pipeline

from nltk.stem.cistem import Cistem

from src.utils.helper_functions_defined_by_user._functions_general import (
    check_dbfs_existence,
)

# Functions originally in _functions_nlp
decode_udf = udf(urlParse.unquote)

charsFrom = (
    "ÀÁÂÃÄÅČÇĎÈÉÊËÍÌÏÎĹĽŇÑÒÓÔÕÖŔŘŠŤÙÚŮÛŰÜÝŸŽàáâãäåčçďèéêëíìïîĺľňñòóôõöŕřšťùúůûűüýÿž"
)
charsTo = (
    "AAAAAACCDEEEEIIIILLNNOOOOORRSTUUUUUUYYZaaaaaaccdeeeeiiiillnnooooorrstuuuuuuyyz"
)


@F.udf(returnType=T.ArrayType(T.StringType()))
def udfstemed_lst(sentence, account=None, aggressive=False, inputString=False):
    # TODO: the stemming is too agresive and not very smart - it should be refactored

    # helper functions (due to issue with import of UDF when helper functions are outside of the function's body)
    def _remove_case(word):
        if len(word) > 7 and word.endswith("atech"):
            return word[:-5]
        if len(word) > 6:
            if word.endswith("ětem"):
                return _palatalise(word[:-3])
            if word.endswith("atům"):
                return word[:-4]
        if len(word) > 5:
            if word[-3:] in {
                "ech",
                "ich",
                "ích",
                "ého",
                "ěmi",
                "emi",
                "ému",
                "ete",
                "eti",
                "iho",
                "ího",
                "ími",
                "imu",
            }:
                return _palatalise(word[:-2])
            if word[-3:] in {
                "ách",
                "ata",
                "aty",
                "ých",
                "ama",
                "ami",
                "ové",
                "ovi",
                "ými",
            }:
                return word[:-3]
        if len(word) > 4:
            if word.endswith("em"):
                return _palatalise(word[:-1])
            if word[-2:] in {"es", "ém", "ím"}:
                return _palatalise(word[:-2])
            if word[-2:] in {"ům", "at", "ám", "os", "us", "ým", "mi", "ou"}:
                return word[:-2]
        if len(word) > 3:
            if word[-1] in "eiíě":
                return _palatalise(word)
            if word[-1] in "uyůaoáéý":
                return word[:-1]
        return word

    def _remove_possessives(word):
        if len(word) > 5:
            if word[-2:] in {"ov", "ův"}:
                return word[:-2]
            if word.endswith("in"):
                return _palatalise(word[:-1])
        return word

    def _remove_comparative(word):
        if len(word) > 5:
            if word[-3:] in {"ejš", "ějš"}:
                return _palatalise(word[:-2])
        return word

    def _remove_diminutive(word):
        # TODO rewrite using match?
        if len(word) > 7 and word.endswith("oušek"):
            return word[:-5]
        if len(word) > 6:
            if word[-4:] in {
                "eček",
                "éček",
                "iček",
                "íček",
                "enek",
                "ének",
                "inek",
                "ínek",
            }:
                return _palatalise(word[:-3])
            if word[-4:] in {
                "áček",
                "aček",
                "oček",
                "uček",
                "anek",
                "onek",
                "unek",
                "ánek",
            }:
                return _palatalise(word[:-4])
        if len(word) > 5:
            if word[-3:] in {"ečk", "éčk", "ičk", "íčk", "enk", "énk", "ink", "ínk"}:
                return _palatalise(word[:-3])
            if word[-3:] in {
                "áčk",
                "ačk",
                "očk",
                "učk",
                "ank",
                "onk",
                "unk",
                "átk",
                "ánk",
                "ušk",
            }:
                return word[:-3]
        if len(word) > 4:
            if word[-2:] in {"ek", "ék", "ík", "ik"}:
                return _palatalise(word[:-1])
            if word[-2:] in {"ák", "ak", "ok", "uk"}:
                return word[:-1]
        if len(word) > 3 and word[-1] == "k":
            return word[:-1]
        return word

    def _remove_augmentative(word):
        if len(word) > 6 and word.endswith("ajzn"):
            return word[:-4]
        if len(word) > 5 and word[-3:] in {"izn", "isk"}:
            return _palatalise(word[:-2])
        if len(word) > 4 and word.endswith("ák"):
            return word[:-2]
        return word

    def _remove_derivational(word):
        if len(word) > 8 and word.endswith("obinec"):
            return word[:-6]
        if len(word) > 7:
            if word.endswith("ionář"):
                return _palatalise(word[:-4])
            if word[-5:] in {"ovisk", "ovstv", "ovišt", "ovník"}:
                return word[:-5]
        if len(word) > 6:
            if word[-4:] in {
                "ásek",
                "loun",
                "nost",
                "teln",
                "ovec",
                "ovík",
                "ovtv",
                "ovin",
                "štin",
            }:
                return word[:-4]
            if word[-4:] in {"enic", "inec", "itel"}:
                return _palatalise(word[:-3])
        if len(word) > 5:
            if word.endswith("árn"):
                return word[:-3]
            if word[-3:] in {"ěnk", "ián", "ist", "isk", "išt", "itb", "írn"}:
                return _palatalise(word[:-2])
            if word[-3:] in {
                "och",
                "ost",
                "ovn",
                "oun",
                "out",
                "ouš",
                "ušk",
                "kyn",
                "čan",
                "kář",
                "néř",
                "ník",
                "ctv",
                "stv",
            }:
                return word[:-3]
        if len(word) > 4:
            if word[-2:] in {"áč", "ač", "án", "an", "ář", "as"}:
                return word[:-2]
            if word[-2:] in {
                "ec",
                "en",
                "ěn",
                "éř",
                "íř",
                "ic",
                "in",
                "ín",
                "it",
                "iv",
            }:
                return _palatalise(word[:-1])
            if word[-2:] in {
                "ob",
                "ot",
                "ov",
                "oň",
                "ul",
                "yn",
                "čk",
                "čn",
                "dl",
                "nk",
                "tv",
                "tk",
                "vk",
            }:
                return word[:-2]
        if len(word) > 3 and word[-1] in "cčklnt":
            return word[:-1]
        return word

    def _palatalise(word):
        if word[-2:] in {"ci", "ce", "či", "če"}:
            return word[:-2] + "k"

        if word[-2:] in {"zi", "ze", "ži", "že"}:
            return word[:-2] + "h"

        if word[-3:] in {"čtě", "čti", "čtí"}:
            return word[:-3] + "ck"

        if word[-3:] in {"ště", "šti", "ští"}:
            return word[:-3] + "sk"
        return word[:-1]

    # assume the account ends with 'cz' if not specified (None)
    if (account is not None) and account.endswith("_at"):
        print("at stemmer")
        if sentence is None:
            return []
        if isinstance(sentence, str):
            words = sentence.split(" ")
        else:
            words = sentence
        stemmer = Cistem()
        stemmed_words = []

        for word in list(words):
            stemmed_words.append(stemmer.segment(word)[0])

        return stemmed_words

    #   udfstemed_lst = F.udf(cz_stem, T.ArrayType(T.StringType()))
    else:
        print("cz stemmer")

        if inputString:
            words = sentence.split()
            stemmed_sentence = ""
            for word in words:
                # stemmed_word=""
                if not re.match("^\\w+$", word):
                    stemmed_sentence += word + " "
                elif not word.islower() and not word.istitle() and not word.isupper():
                    stemmed_sentence += word + " "
                else:
                    s = word.lower()  # all our pattern matching is done in lowercase
                    s = _remove_case(s)
                    s = _remove_possessives(s)

                    if aggressive:
                        s = _remove_comparative(s)
                        s = _remove_diminutive(s)
                        s = _remove_augmentative(s)
                        s = _remove_derivational(s)
                    if word.isupper():
                        stemmed_sentence += s.upper() + " "
                    elif word.istitle():
                        stemmed_sentence += s.capitalize() + " "
                    else:
                        stemmed_sentence += s + " "
        else:
            words = sentence
            stemmed_sentence = []
            for word in words:
                # stemmed_word=""
                if not re.match("^\\w+$", word):
                    stemmed_sentence.append(word)
                elif not word.islower() and not word.istitle() and not word.isupper():
                    stemmed_sentence.append(word)
                else:
                    s = word.lower()  # all our pattern matching is done in lowercase
                    s = _remove_case(s)
                    s = _remove_possessives(s)

                    if aggressive:
                        s = _remove_comparative(s)
                        s = _remove_diminutive(s)
                        s = _remove_augmentative(s)
                        s = _remove_derivational(s)
                    if word.isupper():
                        stemmed_sentence.append(s.upper())
                    elif word.istitle():
                        stemmed_sentence.append(s.capitalize())
                    else:
                        stemmed_sentence.append(s)
        return stemmed_sentence


def strip_diacritic_fl(kw_l):
    if kw_l is not None:
        return [unidecode(kw) for kw in kw_l]
    else:
        return None


def run_azuresearch_al(
    interest,
    account,
    client_name,
    enhance,
    personas_client_id,
    specific_solution_path,
    spark,
    dbutils,
):
    notebook_name = "WL_generator_augmented"
    min_selection_percentile = "0.75"
    dbutils.notebook.run(
        notebook_name,
        7200,
        {
            "account": account,
            "client_name": client_name,
            "min_selection_percentile": min_selection_percentile,
            "enhance": enhance,
            "interest": interest,
            "personas_client_id": personas_client_id,
        },
    )
    parquet_file_path = specific_solution_path + f"{interest}_DC.parquet"
    if check_dbfs_existence(parquet_file_path, spark):
        return (
            spark.read.parquet(parquet_file_path)
            .select("DomainCategory")
            .withColumn("interest", F.lit(interest))
            .withColumn("score", F.lit(1))
            .withColumn("rank", F.lit(1))
        )
    else:
        raise ValueError(f"table {parquet_file_path} not generated.")


def clean_rtb_domain(df: DataFrame) -> DataFrame:
    """cleaning of RtbDomain for grouping"""
    df = df.withColumn("RtbDomain", F.lower(F.col("URL"))).drop("URL")
    # remove prefix (https/http/www...) from RtbDomain
    df = df.withColumn(
        "RtbDomain",
        F.regexp_replace("RtbDomain", r"^((https?|ftp)://)?(ww+[^\.]*\.)?", ""),
    )
    # remove '.' if it is the first character in RtbDomain
    df = df.withColumn("RtbDomain", F.regexp_replace("RtbDomain", r"^\.", ""))
    # remove forbidden and all following characters from RtbDomain
    df = df.withColumn(
        "RtbDomain",
        F.regexp_replace(
            "RtbDomain", r"[~\!@#\$%\^&\*\(\)=+\\\|\?<>\:'\"\[\]\{\} ,].*$", ""
        ),
    )
    # crop to domain and category (+ a dot is not allowed after the first forward slash)
    df = df.withColumn(
        "RtbDomain", F.regexp_extract("RtbDomain", r"^[^/]+(/[^/.]+)?", 0)
    )

    # crop domain to the first 100 characters
    df = df.withColumn(
        "RtbDomain",
        F.when(
            F.length(F.col("RtbDomain")) > 99, df["RtbDomain"].substr(0, 99)
        ).otherwise(F.col("RtbDomain")),
    )

    # remove trailing special char
    df = df.withColumn("RtbDomain", F.regexp_replace("RtbDomain", r"[\.\-\/]$", ""))

    # remove "m" indicating mobile phone
    df = df.withColumn("RtbDomain", F.regexp_replace("RtbDomain", r"^m\.", ""))
    df = df.withColumn("RtbDomain", F.regexp_replace("RtbDomain", r"\.m\.", "."))
    df = df.withColumnRenamed("RtbDomain", "URL")
    return df


# Functions originally in _NLP_functions
def enhance_keywords(
    kw_list,
    synonym_count,
    account,
    spark: SparkSession,
    sc: SparkContext,
    sql_context: SQLContext,
    max_number=100,
):
    # pylint: disable=protected-access
    synonyms = pd.DataFrame({}, columns=["word", "similarity", "count"])
    if account.endswith("_at"):
        sc._jvm.example.Word2VecGE.synonyms(kw_list, synonym_count)
    else:
        sc._jvm.example.Word2VecCZ.synonyms(kw_list, synonym_count)
    synonyms = sql_context.table("df").withColumn("count", F.lit(1)).toPandas()

    synonyms = (
        synonyms.groupby("word")
        .agg({"similarity": "sum", "count": "count"})
        .sort_values(by=["count", "similarity"], ascending=False)
        .reset_index(drop=False)
    )
    spark.catalog.dropTempView("df")
    return " ".join(list(synonyms["word"].loc[:max_number]))


def logit(groups):
    return [
        (F.round(F.greatest(F.tanh(F.log(0.5 + F.sum(F.col(x)))), F.lit(0)), 5)).alias(
            x
        )
        for x in groups
    ]


def indexesCalculator(data: DataFrame, levelOfDistinction=None) -> DataFrame:
    if levelOfDistinction is None:
        levelOfDistinction = ["DomainCategory"]
    temp = (
        data.select(*(levelOfDistinction + ["interest"]))
        .groupBy(*levelOfDistinction)
        .agg(*logit(groups=["interest"]))
    )
    return temp


def strip_diacritic(s) -> str:
    """
    Strips diacritic from a string.
    First, normalize the string, that is split diacritic ('Mn' category) and letters and then omits 'Mn category'
    """
    if isinstance(s, str) and s != "":
        s_normalized = unicodedata.normalize("NFD", s)
        out_s = "".join(c for c in s_normalized if unicodedata.category(c) != "Mn")
        return out_s
    else:
        return ""


def url_to_adform_format(url: str) -> str:
    """
    Process and crop the domains in input_col_name so that they are in compliance with AdForm format (no special characters,
    maximum length of 100 characters) and also cropped to domain/category.
    """
    if not url:
        return None

    transformed_url = url.lower()
    transformed_url = re.sub(r"^((https?|ftp)://)?", "", transformed_url)
    transformed_url = re.sub(r"^\.", "", transformed_url)
    transformed_url = re.sub(
        r"[~\!@#\$%\^&\*\(\)=+\\\|\?<>\:'\"\[\]\{\} ,].*$", "", transformed_url
    )
    transformed_url = re.sub(r"^m\.", "", transformed_url)
    transformed_url = re.sub(r"\.m\.", ".", transformed_url)
    transformed_url = re.search(r"^[^/]+(/[^/.]+)?", transformed_url).group(0)
    transformed_url = transformed_url[:99]
    transformed_url = re.sub(r"[\.\-\/]$", "", transformed_url)
    transformed_url = re.sub(r"www.", "", transformed_url)
    transformed_url = re.sub(r"m,.", "", transformed_url)
    return transformed_url


def url_to_adform_format_light(url: str) -> str:
    """
    Light version of url_to_adform_format - no cropping and shortening.
    """
    if not url:
        return None
    else:
        transformed_url = url.lower()
        transformed_url = re.sub(r"^((https?|ftp)://)?", "", transformed_url)
        transformed_url = re.sub(r"^\.", "", transformed_url)
        transformed_url = re.sub(
            r"[~\!@#\$%\^&\*\(\)=+\\\|\?<>\:'\"\[\]\{\} ,].*$", "", transformed_url
        )
        transformed_url = re.sub(r"^m\.", "", transformed_url)
        transformed_url = re.sub(r"\.m\.", ".", transformed_url)
        # transformed_url = re.search(r"^[^/]+(/[^/.]+)?", transformed_url).group(0) # no cropping
        # transformed_url = transformed_url[:99] # no length-based trimming
        transformed_url = re.sub(r"[\.\-\/]$", "", transformed_url)
        transformed_url = re.sub(r"www.", "", transformed_url)
        transformed_url = re.sub(r"m,.", "", transformed_url)
        return transformed_url


# spark.udf.register("url_transform_light", url_to_adform_format_light)
url_transform_light = F.udf(url_to_adform_format_light, T.StringType())


def df_url_to_adform_format(df: DataFrame, column: str) -> DataFrame:
    """
    Process and crop the domains in input_col_name so that they are in compliance with AdForm format (no special characters,
    maximum length of 100 characters) and also cropped to domain/category.
    """
    df = df.withColumn(column, F.lower(F.col(column)))
    df = df.withColumn(
        column, F.regexp_replace(F.col(column), r"^((https?|ftp)://)?", "")
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"^\.", ""))
    df = df.withColumn(
        column,
        F.regexp_replace(
            F.col(column), r"[~\!@#\$%\^&\*\(\)=+\\\|\?<>\:'\"\[\]\{\} ,].*$", ""
        ),
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"^m\.", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"\.m\.", "."))
    df = df.withColumn(column, F.regexp_extract(F.col(column), r"^[^/]+(/[^/.]+)?", 0))
    df = df.withColumn(
        column,
        F.when(F.length(F.col(column)) > 99, df[column].substr(0, 99)).otherwise(
            F.col(column)
        ),
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"[\.\-\/]$", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"www.", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"m,.", ""))
    return df


def df_url_to_adform_format_light(df: DataFrame, column: str) -> DataFrame:
    """
    Light version of url_to_adform_format - no cropping and shortening.
    """

    df = df.withColumn(column, F.lower(F.col(column)))
    df = df.withColumn(
        column, F.regexp_replace(F.col(column), r"^((https?|ftp)://)?", "")
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"^\.", ""))
    df = df.withColumn(
        column,
        F.regexp_replace(
            F.col(column), r"[~\!@#\$%\^&\*\(\)=+\\\|\?<>\:'\"\[\]\{\} ,].*$", ""
        ),
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"^m\.", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"\.m\.", "."))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"[\.\-\/]$", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"www.", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"m,.", ""))
    return df


def df_url_to_domain(df: DataFrame, column: str, subdomains=None) -> DataFrame:
    """
    Process and crop the domains in input_col_name so that they are in compliance with AdForm format (no special characters,
    maximum length of 100 characters) and also cropped to domain/category.
    """
    df = df.withColumn(column, F.lower(F.col(column)))
    df = df.withColumn(
        column, F.regexp_replace(F.col(column), r"^((https?|ftp)://)?", "")
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"^\.", ""))
    df = df.withColumn(
        column,
        F.regexp_replace(
            F.col(column), r"[~\!@#\$%\^&\*\(\)=+\\\|\?<>\:'\"\[\]\{\} ,].*$", ""
        ),
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"^m\.", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"\.m\.", "."))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"[\.\-\/]$", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"www.", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"m,.", ""))
    df = df.withColumn(column, F.split(F.col(column), "/").getItem(0))
    if subdomains == 0:
        df = df.withColumn(column, F.reverse(F.split(F.col(column), r"\.")).getItem(0))
    elif subdomains == 1:
        df = df.withColumn(
            column,
            F.concat_ws(
                ".",
                F.reverse(
                    F.slice(F.reverse(F.split(F.col(column), r"\.")), start=1, length=2)
                ),
            ),
        )
    elif subdomains == 2:
        df = df.withColumn(
            column,
            F.concat_ws(
                ".",
                F.reverse(
                    F.slice(F.reverse(F.split(F.col(column), r"\.")), start=1, length=3)
                ),
            ),
        )
    return df


def df_url_to_2_level_domain(df: DataFrame, column: str) -> DataFrame:
    """
    Process and crop the domains in input_col_name so that they are in compliance with AdForm format (no special characters,
    maximum length of 100 characters) and also cropped to domain/category.
    """
    df = df.withColumn(column, F.lower(F.col(column)))
    df = df.withColumn(
        column, F.regexp_replace(F.col(column), r"^((https?|ftp)://)?", "")
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"^\.", ""))
    df = df.withColumn(
        column,
        F.regexp_replace(
            F.col(column), r"[~\!@#\$%\^&\*\(\)=+\\\|\?<>\:'\"\[\]\{\} ,].*$", ""
        ),
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"^m\.", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"\.m\.", "."))
    df = df.withColumn(
        column, F.regexp_extract(F.col(column), r"^[^/]+(/[^/.]+)(/[^/.]+)", 0)
    )
    df = df.withColumn(
        column,
        F.when(F.length(F.col(column)) > 99, df[column].substr(0, 99)).otherwise(
            F.col(column)
        ),
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"[\.\-\/]$", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"www.", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"m,.", ""))
    return df


def df_url_normalization(df: DataFrame, column: str) -> DataFrame:
    """
    Normalization of URL
    """
    df = df.withColumn(column, F.lower(F.col(column)))
    df = df.withColumn(
        column, F.regexp_replace(F.col(column), r"^((https?|ftp)://)?", "")
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"^\.", ""))
    df = df.withColumn(
        column,
        F.regexp_replace(
            F.col(column), r"[~\!@#\$%\^&\*\(\)=+\\\|\?<>\:'\"\[\]\{\} ,].*$", ""
        ),
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"^m\.", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"\.m\.", "."))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r".php$", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r".html$", ""))
    df = df.withColumn(
        column, F.regexp_replace(F.col(column), r"\.[0-9a-zA-z_-]*$", "")
    )
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"^(www.)?", ""))
    df = df.withColumn(column, F.regexp_replace(F.col(column), r"/$", ""))
    return df


def df_url_to_DV360(df: DataFrame) -> DataFrame:
    """
    Function to format URL that can be uploaded to DV360
    Format specifications: - Supports 2 level categories (i.e. mydomain/first-cat/second-cat)
                           - Supports up to 5 levels of subdomain targeting (five.four.three.two.one.mydomain.com)
    Further documentation: https://support.google.com/displayvideo/answer/2650521?hl=en
    """
    # TODO: finish or remove?
    return df


def clean_domain_column(df: DataFrame, column: str) -> DataFrame:
    return (
        # transform urls to lowercase
        df.withColumn(column, F.lower(F.col(column)))
        # remove prefix (https/http...) from RtbDomain
        .withColumn(column, F.regexp_replace(column, r"^((https?|ftp)://)?", ""))
        # remove '.' if it is the first character in RtbDomain
        .withColumn(column, F.regexp_replace(column, r"^\.", ""))
        # remove '/' if it is the last character in RtbDomain
        .withColumn(column, F.regexp_replace(column, r"/$", ""))
        # remove ? and # and all following characters from RtbDomain
        .withColumn(column, F.regexp_replace(column, r"[?#].*$", ""))
        #     .withColumn('RtbDomain', F.regexp_replace("RtbDomain", '[~\!@#\$%\^&\*\(\)=+\\\|\?<>\:\'\"\[\]\{\} ,].*$', ''))
    )


def df_stemming(
    df: DataFrame,
    input_col: str,
    cleaned_col: str,
    stemmed_col: str,
    client_name: str,
    stop_words: list,
) -> DataFrame:
    """
    Performs both cleaning and stemming of a column, creating new column for each transformation (cleaned, cleaned & stemmed)
    """
    # TODO: this function is obsolete and it will not be used after the main pipeline refactoring

    df_augmented = df.filter(F.col(input_col).isNotNull())
    df_null = df.filter(F.col(input_col).isNull())

    # remove stopwords
    tokenizer = RegexTokenizer(
        minTokenLength=2,
        pattern=r"[\W_]+",
        inputCol=input_col,
        outputCol="input_col_temp",
    )
    remover = StopWordsRemover(
        stopWords=stop_words, inputCol="input_col_temp", outputCol=cleaned_col
    )

    pipeline = Pipeline(stages=[tokenizer, remover])
    pipeline_fitted = pipeline.fit(df_augmented)
    df_augmented = pipeline_fitted.transform(df_augmented)

    df_augmented = df_augmented.withColumn(
        stemmed_col, udfstemed_lst(F.col(cleaned_col), F.lit(client_name))
    ).drop("input_col_temp")

    return df_augmented.unionByName(df_null, allowMissingColumns=True)


def crop_to_DC(df_input: DataFrame, inputcolname: str):
    match_2_sections = "".join(["[^/]*/?" for _ in range(2)])

    df_cropped_urls0 = (
        df_input.withColumn(
            "DomainCategory",
            F.regexp_extract(
                F.col(inputcolname), r"^((https?://|ftp://))?" + match_2_sections, 0
            ),
        )
        .withColumn(
            "DomainCategory",
            F.regexp_replace(
                F.col("DomainCategory"),
                r"^((https?|ftp)://)?(ww+[^\.]*\.)?(\*\.)?(m\.)?(mobil\.)?(3c\.)?",
                "",
            ),
        )
        .withColumn(
            "DomainCategory", F.regexp_replace(F.col("DomainCategory"), r"/#?$", "")
        )
    )

    # remove forbidden and all following characters from domain
    df_cropped_urls = df_cropped_urls0.withColumn(
        "DomainCategory",
        F.regexp_replace(
            "DomainCategory", r"[~\!@#\$%\^&\*\(\)=+\\\|\?<>\:'\"\[\]\{\} ,].*$", ""
        ),
    )
    return df_cropped_urls


def url_to_domain(url: str) -> str:
    """
    Process and crop the domains in input_col_name so that they are in compliance with AdForm format (no special characters,
    maximum length of 100 characters) and also cropped to domain/category.
    """
    if not url:
        return None
    else:
        transformed_url = url.lower()
        transformed_url = re.sub(r"^((https?|ftp)://)?", "", transformed_url)
        transformed_url = re.sub(r"^\.", "", transformed_url)
        transformed_url = re.sub(
            r"[~\!@#\$%\^&\*\(\)=+\\\|\?<>\:'\"\[\]\{\} ,].*$", "", transformed_url
        )
        transformed_url = re.sub(r"^m\.", "", transformed_url)
        transformed_url = re.sub(r"\.m\.", ".", transformed_url)
        transformed_url = re.sub(r"[\.\-\/]$", "", transformed_url)
        transformed_url = re.sub(r"www.", "", transformed_url)
        transformed_url = re.sub(r"m,.", "", transformed_url)
        return transformed_url.split("/")[0]


# helper to process string to list (for widget input)
def convert_string_to_list(string: str) -> list:
    # pylint: disable=consider-using-in
    if string == "[]" or string == "":
        return []
    else:
        string = re.sub("[\\[\\]]", "", string)
        return list(string.split(", "))
