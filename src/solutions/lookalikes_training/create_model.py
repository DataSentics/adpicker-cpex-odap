# Databricks notebook source
# MAGIC %run ./../../app/bootstrap

# COMMAND ----------

import logging
import json
from functools import partial, reduce
from sys import version_info

import mlflow
import pandas as pd
import scipy
from hyperopt import fmin, tpe, hp, space_eval
from hyperopt.pyll import scope

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import types as T
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import (
    VectorAssembler,
    StringIndexer,
    StandardScaler,
    OneHotEncoder, )
from pyspark.sql import DataFrame as D
from pyspark.sql.window import Window

import daipe as dp
from adpickercpex.solutions.lookalikes_training.preprocessing import IGNORED_COLS, interpolate_unknowns, \
    get_relevant_colnames
from adpickercpex.utils.monitoring.monitoring_functions import calculate_lift, fi_plot

# COMMAND ----------

# MAGIC %md
# MAGIC #### INIT:

# COMMAND ----------

logger = logging.getLogger("py4j")
logger.setLevel(logging.ERROR)
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

# for key initialization
@dp.notebook_function()
def get_interest_list(fs: dp.fs.FeatureStore):
    lst = fs.get_metadata()
    return lst

# COMMAND ----------

dbutils.widgets.text("train_set_location", "train_set_location")
data_name = dbutils.widgets.get("train_set_location")

dbutils.widgets.text("segment_id", "segment_id")
segment_id = dbutils.widgets.get("segment_id")

# COMMAND ----------

# MAGIC %md
# MAGIC #### CONSTANTS:

# COMMAND ----------

conda_env = {
    "channels"    : ['defaults', 'conda-forge'],
    "dependencies": [
        f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
        "pip",
        {"pip": [f"mlflow=={mlflow.__version__}",
                 f"pandas=={pd.__version__}",
                 f"pyspark=={pyspark.__version__}",
                 f"scipy=={scipy.__version__}",
                 ]
         },
    ],
    "name"        : "mlflow-env"
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read data

# COMMAND ----------

fs = spark.table(data_name)
fs = fs.drop(*IGNORED_COLS)
fs = interpolate_unknowns(fs)

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC # HYPERPARAMETERS

# COMMAND ----------

hyperparam_common = {
    "sample_neg_class": scope.float(hp.quniform("sample_neg_class", 0.1, 0.7, q=0.1)),
    "sample_pos_class": scope.float(hp.quniform("sample_pos_class", 1, 10, q=1))
}

hyperparam_space_rf = {
                          "numTrees"           : scope.int(hp.quniform("numTrees", 50, 400, q=10)),
                          "maxDepth"           : scope.int(hp.quniform("maxDepth", 3, 8, q=1)),
                          "minInstancesPerNode": scope.int(hp.quniform("minInstancesPerNode", 20, 100, q=10)),
                          "subsamplingRate"    : scope.float(hp.quniform("subsamplingRate", 0.1, 1, q=0.1))
                      } | hyperparam_common

hyperparam_space_lr = {
                          "regParam"       : hp.choice("regParam", [0.01, 0.001, 0.0001]),
                          "elasticNetParam": hp.choice("elasticNetParam", [0.0, 1.0]),
                      } | hyperparam_common

# COMMAND ----------

# MAGIC %md
# MAGIC # POST-FITTING FUNCTIONS:

# COMMAND ----------

def make_predictions(model_best, df, cutoff=0.9):
    def ith_(v, i):
        try:
            return float(v[i])
        except ValueError:
            return None

    ith = udf(ith_, T.DoubleType())

    wind = Window.orderBy(F.col("pred_prob"))
    df_preds = (model_best.transform(df)
                .withColumn("pred_prob", ith(F.col("probability"), F.lit(1)))
                .withColumn("prob_percentile", F.percent_rank().over(wind))
                .withColumn("label_predicted", F.when(F.col("prob_percentile") >= cutoff, 1).otherwise(0)))
    return df_preds

# COMMAND ----------

# MAGIC %md
# MAGIC # FITTING FUNCTIONS:

# COMMAND ----------

def resample(df, required_pos_ratio, required_neg_ratio, label, pos_class):
    replac_pos = False
    if required_pos_ratio > 1:
        replac_pos = True

    replac_neg = False
    if required_neg_ratio > 1:
        replac_neg = True

    pos = df.filter(F.col(label) == pos_class)
    neg = df.filter(F.col(label) != pos_class)

    sampled_neg = neg.sample(replac_neg, required_neg_ratio)
    sampled_pos = pos.sample(replac_pos, required_pos_ratio)

    total_pos = pos.count()
    total_neg = neg.count()
    total_sampled_pos = sampled_pos.count()
    total_sampled_neg = sampled_neg.count()

    info_dict = {"total_pos"        : total_pos,
                 "total_neg"        : total_neg,
                 "total_sampled_pos": total_sampled_pos,
                 "total_sampled_neg": total_sampled_neg}

    return sampled_neg.union(sampled_pos), info_dict

# COMMAND ----------

def train_RF_classifier(params, train_val_df):
    estimator = RandomForestClassifier(
            numTrees=params["numTrees"],
            maxDepth=params["maxDepth"],
            minInstancesPerNode=params["minInstancesPerNode"],
            subsamplingRate=params["subsamplingRate"],
            featuresCol="features",
            labelCol="label")
    mlflow.log_param("maxDepth", params["maxDepth"])
    mlflow.log_param("numTrees", params["numTrees"])
    mlflow.log_param("minInstancesPerNode", params["minInstancesPerNode"])
    mlflow.log_param("subsamplingRate", params["subsamplingRate"])

    numerical_colnames, categorical_colnames, cat_cols_indexed, cat_cols_encoded = get_relevant_colnames(train_val_df)
    pipe = Pipeline(stages=[StringIndexer(inputCols=categorical_colnames,
                                          outputCols=cat_cols_indexed, handleInvalid="skip"),
                            VectorAssembler(inputCols=cat_cols_indexed + numerical_colnames,
                                            outputCol="features", handleInvalid="skip"),
                            estimator])
    model_pipe = pipe.fit(train_val_df)
    fig = fi_plot(model_pipe)
    mlflow.log_figure(fig, "feature_importances.png")
    return model_pipe

# COMMAND ----------

def train_LR(params, train_val_df):
    estimator = LogisticRegression(
            regParam=params["regParam"],
            elasticNetParam=params["elasticNetParam"],
            featuresCol="features",
            labelCol="label")
    mlflow.log_param("regParam", params["regParam"])
    mlflow.log_param("elasticNetParam", params["elasticNetParam"])

    numerical_colnames, categorical_colnames, cat_cols_indexed, cat_cols_encoded = get_relevant_colnames(train_val_df)
    pipe = Pipeline(stages=[StringIndexer(inputCols=categorical_colnames,
                                          outputCols=cat_cols_indexed, handleInvalid="skip"),
                            OneHotEncoder(inputCols=cat_cols_indexed,
                                          outputCols=cat_cols_encoded),
                            VectorAssembler(inputCols=cat_cols_encoded + numerical_colnames,
                                            outputCol="features_assembled", handleInvalid="keep",
                                            ),
                            StandardScaler(inputCol="features_assembled",
                                           outputCol="features_scaled", withMean=True, withStd=True),
                            VectorAssembler(inputCols=["features_scaled"],
                                            outputCol="features"),
                            estimator])
    model_pipe = pipe.fit(train_val_df)
    return model_pipe

# COMMAND ----------

def fit_model(params, modeling_method, train_val_df):
    required_neg_ratio = params["sample_neg_class"]
    required_pos_ratio = params["sample_pos_class"]
    train_val_df, info_dict = resample(train_val_df, required_pos_ratio, required_neg_ratio, "label", 1)
    mlflow.log_param("total_pos_label", info_dict["total_pos"])
    mlflow.log_param("total_neg_label", info_dict["total_neg"])
    mlflow.log_param("total_sampled_pos_label", info_dict["total_sampled_pos"])
    mlflow.log_param("total_sampled_neg_label", info_dict["total_sampled_neg"])

    if modeling_method == "RF":
        model_pipe = train_RF_classifier(params, train_val_df)

    elif modeling_method == "LR":
        model_pipe = train_LR(params, train_val_df)
        coefficients = model_pipe.stages[5].coefficients.toArray().tolist()
        with open("/tmp/coefficients.txt", "w") as out_file:
            for item in coefficients:
                out_file.write(str(item) + "\n")
        mlflow.log_artifact("/tmp/coefficients.txt")
    else:
        raise NotImplementedError(f"{modeling_method=} is not implemented use LR (linear regression) or RF (random forest)")

    mlflow.spark.log_model(model_pipe, "model_pipe", conda_env=conda_env)
    return model_pipe

# COMMAND ----------

def objective(params, df, modeling_method):
    total_val_lift = 0
    sample1, sample2, sample3 = df.randomSplit([0.3333, 0.3333, 0.3333])
    samples_list = [sample1, sample2, sample3]
    # stratified fold in spark.ml is hard, this is good enough

    for e, sample in enumerate(samples_list):
        with mlflow.start_run(tags={"segment_id": segment_id,
                                    "method"    : modeling_method},
                              nested=True):
            df_list = samples_list.copy()
            test_val_df = samples_list[e]
            df_list.remove(samples_list[e])
            train_val_df = reduce(D.unionAll, df_list)

            model_pipe = fit_model(params, modeling_method, train_val_df)

            # Predictions
            train_df_preds = make_predictions(model_pipe, train_val_df)
            test_df_preds = make_predictions(model_pipe, test_val_df)

            # calculate lift
            lift_val_train = calculate_lift(train_df_preds)
            lift_val_test = calculate_lift(test_df_preds)
            mlflow.log_metric("lift_val_train", lift_val_train)
            mlflow.log_metric("lift_val_test", lift_val_test)

        total_val_lift += lift_val_test

    return -total_val_lift / 3

# COMMAND ----------

# MAGIC %md
# MAGIC # MAIN FUNCTIONS:

# COMMAND ----------

def find_best_model(fs, max_evals=12):
    mlflow.set_experiment("/adpicker/models/experiment_lookalikes")
    fmin_objective = partial(objective, df=fs, modeling_method="RF")
    # fmin_objective :: parameters -> -val_lift
    with mlflow.start_run(run_name=f"RFC_predictive_features",
                        tags={"segment_id": segment_id,
                            "method"    : "RF"}) as run_rf:
        best = fmin(
                fn=fmin_objective,
                space=hyperparam_space_rf,
                algo=tpe.suggest,
                max_evals=max_evals)
        best_hyperparameters = space_eval(hyperparam_space_rf, best)

        # Train best model on whole training dataset
        model_best_rf = fit_model(best_hyperparameters, "RF", fs)
        mlflow.spark.log_model(model_best_rf, "best_model")

        # Predictions
        preds_rf = make_predictions(model_best_rf, fs)
        lift_rf = calculate_lift(preds_rf)
        mlflow.log_metric("lift_rf", lift_rf)

        # Feature importances
        fig = fi_plot(model_best_rf)
        mlflow.log_figure(fig, "feature_importances.png")


    with mlflow.start_run(run_name=f"LRC_predictive_features",
                          tags={"segment_id": segment_id,
                                "method"    : "LR"
                                }) as run_lr:
        fmin_objective = partial(objective, modeling_method="LR", df=fs)
        best = fmin(
                fn=fmin_objective,
                space=hyperparam_space_lr,
                algo=tpe.suggest,
                max_evals=max_evals)
        best_hyperparameters = space_eval(hyperparam_space_lr, best)

        # Train best model on whole training dataset
        model_best_lr = fit_model(best_hyperparameters, "LR", fs)
        mlflow.spark.log_model(model_best_lr, "best_model", conda_env=conda_env)

        coefficients = model_best_lr.stages[5].coefficients.toArray().tolist()
        # hardcoded 5 to extract model from static pipeline.
        with open("/tmp/coefficients.txt", "w") as out_file:
            for item in coefficients:
                out_file.write(str(item) + "\n")
        mlflow.log_artifact("/tmp/coefficients.txt")

        # Predictions
        preds_lr = make_predictions(model_best_lr, fs)
        lift_lr = calculate_lift(preds_lr)
        mlflow.log_metric("lift_lr", lift_lr)

    if lift_rf > lift_lr:
        run_id = run_rf.info.run_id
    else:
        run_id = run_lr.info.run_id

    model_uri = f"runs:/{run_id}/best_model"
    model_name = f"lookalike_model_{segment_id}"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    return model_name, model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC # Run training

# COMMAND ----------

model_name, model_run_uri = find_best_model(fs, max_evals=12)

model_registry_uri = f"models:/{model_name}/"

model_info = {
    "mlf_model": model_registry_uri,
    "mlf_run"  : model_run_uri
}
print(f"{model_info=}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Save to lookalikes table

# COMMAND ----------

@dp.transformation(dp.read_delta("%lookalike.delta_path%"),
                   segment_id,
                   model_info,
                   display=False)
def update_model_info(lal_df: DataFrame, segment_id: str, model_info):
    return (lal_df.withColumn("Model",
                              F.when(F.col("TP_DMP_id") == segment_id,
                                     json.dumps(model_info))
                              .otherwise(F.col("Model"))))

# COMMAND ----------

@dp.transformation(update_model_info, display=False)
@dp.delta_overwrite("%lookalike.delta_path%")
def update_interests(df):
    return df
