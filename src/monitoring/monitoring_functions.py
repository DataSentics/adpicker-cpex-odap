from functools import reduce
from logging import Logger
from operator import or_

import pandas as pd
import pyspark.sql.functions as F
from matplotlib import pyplot as plt
from pyspark.ml.linalg import DenseVector
from pyspark.sql.dataframe import DataFrame

from src.utils.helper_functions_defined_by_user._functions_ml import lift_curve_with_prob_column


def monitor_n_records_with_target(df: DataFrame, target_col: str, logger: Logger):
    n_records = (df
                    .filter(F.col(target_col).isNotNull())
                    .count()
                )
    logger.info(f"Records with target {target_col}: {n_records}")
    return {target_col: n_records}


def monitor_target_distribution(df: DataFrame, target_col: str, logger: Logger, model_name):
    n_total = (df
                .filter(F.col(target_col).isNotNull())
                .count()
                )

    target_dist = (df
                    .filter(F.col(target_col).isNotNull())
                    .groupBy(target_col)
                    .count()
                    .withColumn('share', F.col('count')/F.lit(n_total))
                    .sort(target_col)
                    )

    logger.info(f'model {model_name}:')
    target_dist.show()
    return reduce(or_,
                    ({f"{model_name}_{row[target_col]}_count": row["count"]} |
                        {f"{model_name}_{row[target_col]}_share": row["share"]}
                    for row in target_dist.collect())
                    )



def monitor_missing_predictions(df: DataFrame, pred_col: str, logger: Logger, model_name):
    # total records
    n_total = df.count()
    n_miss = df.filter(F.col(pred_col).isNull() | (F.col(pred_col) < 0)).count()
    # log result
    logger.info(f"Share of missing predictions for {pred_col}: {n_miss/n_total}")
    return {f"{model_name}_missing_share": n_miss/n_total,
            f"{model_name}_missing_count": n_miss}


def monitor_performance_lift(df: DataFrame, target_col: str, pred_col: str, logger: Logger, model_name):
    df = df.filter(F.col(target_col).isNotNull())

    # calculate lift
    df_lift = lift_curve_with_prob_column(df_predictions=df, target=target_col, prediction=pred_col, bin_count=10)
    lift_first_bin = df_lift.filter(F.col("bucket") == 1).select("cum_lift").collect()[0][0]

    # log result
    logger.info(f"Lift for model {model_name}: {lift_first_bin}")
    return {f"{model_name}_lift_1": lift_first_bin}


def calculate_lift(df_preds, cutoff=0.9):
    positive_class_total = df_preds.filter(F.col("label") == 1).count()
    expected_positive_class = max(round(positive_class_total * (1 - cutoff)), 1)
    actual_positive_class = df_preds.filter((F.col("label") == 1) & (F.col("prob_percentile") >= cutoff)).count()
    lift = round(actual_positive_class / expected_positive_class, 3)
    return lift


def fi_plot(model_rf):
    """ Creates feature importance plots.
    """
    input_cols = model_rf.stages[1].getInputCols()
    feat_importances = DenseVector(model_rf.stages[2].featureImportances.toArray()).values.tolist()
    dct = {"cols": input_cols, "importances": feat_importances}
    pd_importances = pd.DataFrame(dct).sort_values("importances", ascending=False).head(40)
    fig, _ = plt.subplots(figsize=(10, 15))
    plt.barh(pd_importances["cols"][::-1], pd_importances["importances"][::-1])
    plt.title("Feature Importances")
    return fig
