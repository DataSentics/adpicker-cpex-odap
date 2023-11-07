from pyspark.sql.functions import udf
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window
from pyspark.sql.session import SparkSession
from pyspark.sql import DataFrame

import datetime as dt

from typing import Callable, List, Any
import os
from itertools import chain

from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.classification import (
    LogisticRegression,
    GBTClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    MultilayerPerceptronClassifier,
    LinearSVC,
    NaiveBayes,
)

import mlflow
import mlflow.spark as mlflow_spark
from mlflow.tracking.client import MlflowClient


def logit(groups, idfV):
    #####INTERESTS LOGIT VARIABLE: IDF
    minIDF = idfV.values.min()
    totaldif = idfV.values.max() - minIDF
    return [
        (
            F.round(
                (
                    1
                    / (
                        1
                        + F.exp(
                            (
                                -(
                                    0.25
                                    + 0.5
                                    * ((idfV[x][0] - minIDF) / totaldif)
                                    * F.sum(F.col(x))
                                )
                                + 4 * F.exp(-(F.sum(F.col(x))))
                            )
                        )
                    )
                ),
                5,
            )
            - 0.02298
        ).alias(x)
        for x in groups
    ]


def indexesCalculator(data, interests, idfV, levelOfDistinction=None):
    if levelOfDistinction is None:
        levelOfDistinction = ["DomainCategory"]
    temp = (
        data.select(*(levelOfDistinction + interests))
        .groupBy(*levelOfDistinction)
        .agg(*logit(groups=interests, idfV=idfV))
    )
    return temp


def lift_jlh(dataset, column, power=1, minDF=30):
    count_users = dataset.select(column).distinct().count()
    count_users_click = (
        dataset.select(column).filter(F.col("target") == 1).distinct().count()
    )
    dataset = dataset.withColumn("hits", F.explode(F.col("tokens")))

    background_set = (
        dataset.groupby("hits")
        .agg(F.countDistinct(column).alias("token_users"))
        .withColumn("background_pc", F.col("token_users") / count_users * 100)
    )

    foreground_set = (
        dataset.filter(F.col("target") == 1)
        .groupby("hits")
        .agg(F.countDistinct(column).alias("token_users_click"))
        .withColumn(
            "foreground_pc", F.col("token_users_click") / count_users_click * 100
        )
    )

    jlh_set = (
        background_set.join(foreground_set, "hits", "outer")
        .filter(F.col("token_users") >= minDF)
        .fillna(0)
        .select("hits", "token_users", "foreground_pc", "background_pc")
    )

    jlh_DF = (
        jlh_set.withColumn(
            "lift", pow(F.col("foreground_pc") / F.col("background_pc"), power)
        )
        .withColumn("diff", F.col("foreground_pc") - F.col("background_pc"))
        .withColumn("JLH_score", F.col("lift") * F.col("diff"))
        .sort(F.col("JLH_score").desc())
    )

    return jlh_DF


# LIFT
def lift_curve(predictions, target, bin_count):
    vectorElement = udf(lambda v: float(v[1]))
    lift_df = (
        predictions.select(
            vectorElement("category_affinity").cast("float").alias("category_affinity"),
            target,
        )
        .withColumn(
            "rank", F.ntile(bin_count).over(Window.orderBy(F.desc("category_affinity")))
        )
        .select("category_affinity", "rank", target)
        .groupBy("rank")
        .agg(
            F.count(target).alias("bucket_row_number"),
            F.sum(target).alias("bucket_lead_number"),
            F.avg("category_affinity").alias("avg_model_lead_probability"),
        )
        .withColumn(
            "cum_avg_leads",
            F.avg("bucket_lead_number").over(
                Window.orderBy("rank").rangeBetween(Window.unboundedPreceding, 0)
            ),
        )
    )

    avg_lead_rate = (
        lift_df.filter(F.col("rank") == bin_count)
        .select("cum_avg_leads")
        .collect()[0]
        .cum_avg_leads
    )  # cislo = cum_avg_leads 10. decilu napr(317.2)

    cum_lift_df = lift_df.withColumn(
        "cum_lift", F.col("cum_avg_leads").cast("float") / avg_lead_rate
    ).selectExpr(
        "rank as bucket",
        "bucket_row_number",
        "bucket_lead_number",
        "avg_model_lead_probability",
        "cum_avg_leads",
        "cum_lift",
    )
    return cum_lift_df


# LIFT
def lift_curve_colname_specified(predictions, target, bin_count, colname):
    vectorElement = udf(lambda v: float(v[1]))
    lift_df = (
        predictions.select(vectorElement(colname).cast("float").alias(colname), target)
        .withColumn("rank", F.ntile(bin_count).over(Window.orderBy(F.desc(colname))))
        .select(colname, "rank", target)
        .groupBy("rank")
        .agg(
            F.count(target).alias("bucket_row_number"),
            F.sum(target).alias("bucket_lead_number"),
            F.avg(colname).alias("avg_model_lead_probability"),
        )
        .withColumn(
            "cum_avg_leads",
            F.avg("bucket_lead_number").over(
                Window.orderBy("rank").rangeBetween(Window.unboundedPreceding, 0)
            ),
        )
    )

    avg_lead_rate = (
        lift_df.filter(F.col("rank") == bin_count)
        .select("cum_avg_leads")
        .collect()[0]
        .cum_avg_leads
    )  # cislo = cum_avg_leads 10. decilu napr(317.2)

    cum_lift_df = lift_df.withColumn(
        "cum_lift", F.col("cum_avg_leads").cast("float") / avg_lead_rate
    ).selectExpr(
        "rank as bucket",
        "bucket_row_number",
        "bucket_lead_number",
        "avg_model_lead_probability",
        "cum_avg_leads",
        "cum_lift",
    )
    return cum_lift_df


# LIFT - with extracted probability column (i.e. no need to extract vectorElement)
def lift_curve_with_prob_column(df_predictions, target, prediction, bin_count=10):
    lift_df = (
        df_predictions.select(target, prediction)
        .withColumn("rank", F.ntile(bin_count).over(Window.orderBy(F.desc(prediction))))
        .groupBy("rank")
        .agg(
            F.count(target).alias("bucket_row_number"),
            F.sum(target).alias("bucket_lead_number"),
            F.avg(prediction).alias("avg_model_lead_probability"),
        )
        .withColumn(
            "cum_avg_leads",
            F.avg("bucket_lead_number").over(
                Window.orderBy("rank").rangeBetween(Window.unboundedPreceding, 0)
            ),
        )
    )

    avg_lead_rate = (
        lift_df.filter(F.col("rank") == bin_count)
        .select("cum_avg_leads")
        .collect()[0]
        .cum_avg_leads
    )

    return lift_df.withColumn(
        "cum_lift", F.col("cum_avg_leads").cast("float") / avg_lead_rate
    ).selectExpr(
        "rank as bucket",
        "bucket_row_number",
        "bucket_lead_number",
        "avg_model_lead_probability",
        "cum_avg_leads",
        "cum_lift",
    )


# split score
@F.udf(returnType=T.DoubleType())
def ith(v, i):
    try:
        return float(v[i])
    except ValueError:
        return None


# lift words for quadrants
def lift_a_hits(dataset, quadr):
    # drop words with low ocurance
    dataset_higher_occurance = (
        dataset.groupBy("hits")
        .agg(F.count("hits").alias("occurance"))
        .filter(F.col("occurance") > 10)
    )
    dataset_common_words = dataset.join(dataset_higher_occurance, ["hits"], how="inner")

    dataset = dataset_common_words.withColumn(
        "quadrant_0_1", F.when(F.col("quadrant") == quadr, 1).otherwise(0)
    )

    avg_target_rate = dataset.groupBy().agg(F.avg("quadrant_0_1")).collect()[0][0]

    avg_rate_dataset = (
        dataset.groupBy("hits")
        .agg(
            F.avg("quadrant_0_1").alias("hitrate"),
            F.sum("quadrant_0_1").alias("occurance_in"),
            F.count("quadrant_0_1").alias("occurance_all"),
        )
        .withColumn("global_hitrate", F.lit(avg_target_rate))
    )
    avg_rate_dataset = (
        avg_rate_dataset.withColumn("direction", F.col("hitrate") - avg_target_rate)
        .withColumn("improvement", F.col("hitrate") / avg_target_rate)
        .withColumn(
            "impro_times_occurance", F.col("occurance_in") * F.col("improvement")
        )
    )
    avg_rate_dataset = avg_rate_dataset.orderBy(F.desc("impro_times_occurance"))

    return avg_rate_dataset


def fudf(val):
    return F.reduce(lambda x, y: x + y, val)


def column_add(a, b):
    # pylint: disable=unnecessary-dunder-call  # I would rather keep explicit call, bu also what is this for?
    return a.__add__(b)


def load_dataframes(
    accounts: List[str], load_function: Callable[[str], Any]
) -> DataFrame:
    df = load_function(accounts[0])

    for account in accounts[1:]:
        df = df.union(load_function(account))

    return df


def load_trackingpoints(account: str) -> DataFrame:
    return spark.read.table("pl0_all_" + account + ".trackingpoint").filter(
        F.col("CookieID") != "0"
    )


def load_impressions(account: str) -> DataFrame:
    return spark.read.table("pl0_all_" + account + ".impression").filter(
        F.col("CookieID") != "0"
    )


def load_geoloc(client_name):
    return (
        spark.read.table("pl0_all_" + client_name + ".metadata_geolocations")
        .filter(F.col("country").isin(["Slovakia (slovak Republic)"]))
        .withColumnRenamed("cityId", "CityId")
        .select("CityId", "country")
    )


## function to create key, value pair for attribute name and attribute value
def flatten_lists(list_of_list):
    return list(chain(*list_of_list))


##Convert ios_users interest to percentages
def devices(x):
    # '131', '132', '138' - ios devices
    try:
        return len(list(filter(lambda fn: fn in ["131", "132", "138"], x))) / len(x)
    except:
        return 0


devices_udf = udf(devices, T.DoubleType())


def slovak_geolocation(x):
    try:
        return len(
            list(filter(lambda fn: fn in ["Slovakia (slovak Republic)"], x))
        ) / len(x)
    except:
        return 0


slovak_geolocation_udf = udf(slovak_geolocation, T.DoubleType())


def remove_general_urls(df: DataFrame) -> DataFrame:
    GENERAL_URLS = [
        "login.szn",
        "email.seznam.cz",
        "seznam.cz",
        "seznamzpravy.cz/clanek",
        "zkouknito.cz",
        "super.cz",
        "extra.cz",
        "blesk.cz",
        "novinky.cz",
    ]
    return df.where(~(F.col("URL").rlike("|".join(GENERAL_URLS))))


formatToDate = udf(lambda x: dt.datetime.strptime(str(x), "%Y%m%d"), T.DateType())


def process_multiple_segments_input(t):
    t = t.replace(" ", "")

    t_list = list(t.split(","))

    t_table_name = t.replace(",", "_")

    return {
        "converted_list": t_list,
        "table_name_suffix": t_table_name,
        "db_name": t,
    }


# sociodemo
def lift_curve(predictions, target, bin_count=10):
    """Lift curve approximation for binary classification model.

    Function for calculating lift of your binary classification model.
    Designed for standard model `prediction` output as given by LogisticRegressionModel, RandomForestRegressor, etc.
    Data is ordered according to predicted probability from highest
    to lowest and split into `bin_count` bins - 1st bin contains subjects with highest probability.
    Cummulative lift and related metrics are then calculated

    :param predictions: spark data frame, df with 'probability' column - in standard output
    format of LogisticRegressionModel etc. - and target column - contains `1` or `0`, name specified by `target` param
    :param target: string, name of the target column
    :param bin_count: positive int, number of bins you want the data to be split into

    :return: spark dataframe with columns:
        `bucket` - bucket number, bucket with lowest number contains subjects with highest predicted probability
        `bucket_n_subj` - number of subjects in corresponding bucket
        `bucket_n_target` - number of subjects with `target == 1` in corresponding bucket
        `avg_model_target_probability` - average predicted probability for subjects in corresponding bucket
        `cum_avg_target` - average num of subjects with `target == 1` in buckets with number
        lesser or equal to the corresponding bucket number merget together
        `cum_lift` - lift metric for buckets with number lesser or equal to the corresponding
        bucket number merget together, calculated as `cum_avg_target/avg_target_rate`,
        where `avg_target_rate =  sum(bucket_n_subj)/sum(bucket_n_target)`
    """
    if bin_count >= 0 and isinstance(bin_count, int):
        pass
    else:
        raise Exception("Invalid 'bin_count' param value! Set integer value >=0!")

    vectorElement = udf(lambda v: float(v[1]), T.DoubleType())

    lift_df = (
        predictions.select(
            vectorElement("probability").cast("float").alias("probability"), target
        )
        .withColumn(
            "rank", F.ntile(bin_count).over(Window.orderBy(F.desc("probability")))
        )
        .select("probability", "rank", target)
        .groupBy("rank")
        .agg(
            F.count(target).alias("bucket_n_subj"),
            F.sum(target).alias("bucket_n_target"),
            F.avg("probability").alias("avg_model_target_probability"),
        )
        .withColumn(
            "cum_avg_target",
            F.avg("bucket_n_target").over(
                Window.orderBy("rank").rangeBetween(Window.unboundedPreceding, 0)
            ),
        )
    )

    avg_lead_rate = (
        lift_df.filter(F.col("rank") == bin_count)
        .select("cum_avg_target")
        .collect()[0][0]
    )

    cum_lift_df = lift_df.withColumn(
        "cum_lift", F.col("cum_avg_target").cast("float") / avg_lead_rate
    ).selectExpr(
        "rank as bucket",
        "bucket_n_subj",
        "bucket_n_target",
        "avg_model_target_probability",
        "cum_avg_target",
        "cum_lift",
    )
    return cum_lift_df


def lift_curve_generalized(predictions, target, pos_label=1, bin_count=10):
    """Lift curve approximation for binary classification model.

    Function for calculating lift of your binary classification model.
    Designed for standard model `prediction` output as given by LogisticRegressionModel, RandomForestRegressor, etc.
    Data is ordered according to predicted probability from highest to lowest
    and split into `bin_count` bins - 1st bin contains subjects with highest probability.
    Cummulative lift and related metrics are then calculated

    :param predictions: spark data frame, df with 'probability' column - in standard output
    format of LogisticRegressionModel etc. - and target column - contains labels (0,1,...) , name specified by `target` param
    :param target: string, name of the target column
    :param pos_label: int/string? which value of the target column are we interested in
    :param bin_count: positive int, number of bins you want the data to be split into

    :return: spark dataframe with columns:
        `bucket` - bucket number, bucket with lowest number contains subjects with highest predicted probability
        `bucket_n_subj` - number of subjects in corresponding bucket
        `bucket_n_target` - number of subjects with `target == 1` in corresponding bucket
        `avg_model_target_probability` - average predicted probability for subjects in corresponding bucket
        `cum_avg_target` - average num of subjects with `target == 1` in buckets
        with number lesser or equal to the corresponding bucket number merget together
        `cum_lift` - lift metric for buckets with number lesser or equal
        to the corresponding bucket number merget together,
        calculated as `cum_avg_target/avg_target_rate`, where `avg_target_rate =  sum(bucket_n_subj)/sum(bucket_n_target)`
    """
    # this function extracts all probabilities and convert them to list (possible space of improvement if it is not al ist but dictionary)
    vectorToList = udf(
        lambda v: [float(item) for item in v], T.ArrayType(T.DoubleType())
    )

    # input check
    if bin_count < 0 or not isinstance(bin_count, int):
        raise Exception("Invalid 'bin_count' param value! Set integer value >=0!")

    # actual code
    lift_df = (
        predictions.withColumn(
            "probability_list", vectorToList(F.col("probability"))
        )  # extraction of list of probabilites
        .select(
            F.col("probability_list").getItem(pos_label).alias("probability"), target
        )  # extraction of specific probability we are interested in
        .withColumn(
            "rank", F.ntile(bin_count).over(Window.orderBy(F.desc("probability")))
        )  # binning in the bin_count bins
        # .select('probability', 'rank', target)
        .groupBy("rank")
        .agg(
            F.count(target).alias("bucket_n_subj"),
            F.sum(F.when(F.col(target) == pos_label, 1)).alias(
                "bucket_n_target"
            ),  ##counts cases when target == pos_label
            F.avg("probability").alias("avg_model_target_probability"),
        )
        .withColumn(
            "cum_avg_target",
            F.avg("bucket_n_target").over(
                Window.orderBy("rank").rangeBetween(Window.unboundedPreceding, 0)
            ),
        )
    )

    avg_lead_rate = (
        lift_df.filter(F.col("rank") == bin_count)
        .select("cum_avg_target")
        .collect()[0][0]
    )  # what is the total average target

    cum_lift_df = lift_df.withColumn(
        "cum_lift", F.col("cum_avg_target") / avg_lead_rate
    ).selectExpr(
        "rank as bucket",
        "bucket_n_subj",
        "bucket_n_target",
        "avg_model_target_probability",
        "cum_avg_target",
        "cum_lift",
    )
    return cum_lift_df


def compute_lift_train_test(predictions_train, predictions_test, label_column):
    lift_train = (
        lift_curve(predictions_train, label_column, 10)
        .select("bucket", "cum_lift")
        .withColumnRenamed("cum_lift", "lift_train")
    )

    lift_test = (
        lift_curve(predictions_test, label_column, 10)
        .select("bucket", "cum_lift")
        .withColumnRenamed("cum_lift", "lift_test")
    )

    return lift_train.join(lift_test, on="bucket")


def compute_lift_train_test_generalized(
    predictions_train, predictions_test, label_column, pos_label=1, bin_count=10
):
    lift_train = (
        lift_curve_generalized(predictions_train, label_column, pos_label, bin_count)
        .select("bucket", "cum_lift")
        .withColumnRenamed("cum_lift", "lift_train")
    )

    lift_test = (
        lift_curve_generalized(predictions_test, label_column, pos_label, bin_count)
        .select("bucket", "cum_lift")
        .withColumnRenamed("cum_lift", "lift_test")
    )

    return lift_train.join(lift_test, on="bucket").withColumn(
        "pos_label", F.lit(pos_label)
    )


def get_feature_importances(model, feature_columns, spark: SparkSession):
    # if model_type in ['XGBoost', 'RandomForest']:
    feature_importances = list(model.stages[0].featureImportances.toArray())
    feature_importances_with_names = []

    for feature_name, importance in zip(feature_columns, feature_importances):
        feature_importances_with_names.append((feature_name, float(importance)))

    df = spark.createDataFrame(
        feature_importances_with_names, schema="`feature` STRING, `importance` FLOAT"
    )

    return df
