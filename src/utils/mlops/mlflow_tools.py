import logging
import mlflow
from mlflow.exceptions import MlflowException
from contextlib import contextmanager

internal_logger = logging.getLogger(__name__)

viable_stages = ["Production", "Staging", "None"]


def retrieve_model(model_registry_uri, stages, model_version=None, logger=None):
    if logger is None:
        logger = internal_logger
    if model_version is None:
        model_version_str = ""
    else:
        model_version_str = f"/{model_version}"
    if not stages:
        raise ValueError(f"Parameter stages can't be empty, but was {stages}")

    for stage in stages:
        try:
            model_obj = mlflow.spark.load_model(model_registry_uri + stage + model_version_str)
            break
        except MlflowException as e:
            logger.warning(e)
    else:
        if model_version is not None:
            raise MlflowException(f"No model version {model_version} found in {model_registry_uri}, {stages=}")
        raise MlflowException(f"No model found in {model_registry_uri}, {stages=}")

    logger.info("Using model in %s from stage %s",  model_registry_uri, stage)
    return model_obj, stage


@contextmanager
def disable_mlflow_autologging():
    mlflow.autolog(disable=True)
    yield
    mlflow.autolog(disable=False)
