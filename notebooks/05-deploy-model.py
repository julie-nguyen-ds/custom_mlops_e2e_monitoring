# Databricks notebook source
# MAGIC %md
# MAGIC ### Model Deployment
# MAGIC Model deployment job steps:
# MAGIC 1. Compare new “candidate model” in `stage='Staging'` versus current Production model in `stage='Production'`.
# MAGIC 2. Comparison criteria set using f1 score metric.
# MAGIC 3. Compute predictions using both models against a specified reference dataset
# MAGIC   - If Staging model performs better than Production model, promote Staging model to Production and archive existing Production model
# MAGIC   - If Staging model performs worse than Production model, archive Staging model

# COMMAND ----------

from mlflow.tracking import MlflowClient
from databricks.feature_store import FeatureStoreClient

fs_model_name = "fs_home_credit_model"
test_home_credit_table_name: str = (
    "hive_metastore.home_credit_risk.bronze_test_home_credit"
)
label_col: str = "TARGET"

fs = FeatureStoreClient()
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Test Data

# COMMAND ----------

test_credit_data = spark.table(test_home_credit_table_name)
test_credit_data = test_credit_data.withColumn(
    "SK_ID_CURR", test_credit_data.SK_ID_CURR.cast("long")
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Models
# MAGIC We load the model currently in production and the latest model in staging using mlflow.

# COMMAND ----------

# Model currently in production
production_model_version = client.get_latest_versions(
    name=fs_model_name, stages=["production"]
)[0].version
production_uri = f"models:/{fs_model_name}/{production_model_version}"
production_model_version

# COMMAND ----------

# Model that is in staging and might replace production
staging_model_version = client.get_latest_versions(
    name=fs_model_name, stages=["staging"]
)[0].version
staging_uri = f"models:/{fs_model_name}/{staging_model_version}"
staging_model_version

# COMMAND ----------

# MAGIC %md
# MAGIC #### Models Inference
# MAGIC We use each model to score batch inferences on the test dataset and retrieve the predictions results.

# COMMAND ----------

pyfunc_predictions_staging = fs.score_batch(
    staging_uri, test_credit_data, result_type="int"
)
pyfunc_predictions = fs.score_batch(production_uri, test_credit_data, result_type="int")

# COMMAND ----------

display(pyfunc_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Models Scoring
# MAGIC We use each model predictions and compute their respective f1 score.

# COMMAND ----------

import sklearn


def compute_f1_score(df, target_name: str, prediction_name: str):
    pd_df = df.toPandas()
    return sklearn.metrics.f1_score(pd_df[target_name], pd_df[prediction_name])


# COMMAND ----------

prod_score = compute_f1_score(pyfunc_predictions, "TARGET", "prediction")
staging_score = compute_f1_score(pyfunc_predictions_staging, "TARGET", "prediction")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Comparison and Transition

# COMMAND ----------

import logging


class NoReceivedCommandFilter(logging.Filter):
    def filter(self, record):
        if "Received command c" not in record.getMessage():
            return record.getMessage()


class NoPythonDotEnvFilter(logging.Filter):
    def filter(self, record):
        if "Python-dotenv" not in record.getMessage():
            return record.getMessage()


def get_logger():
    logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    filter_1 = NoReceivedCommandFilter()
    filter_2 = NoPythonDotEnvFilter()
    logger.addFilter(filter_1)
    logger.addFilter(filter_2)

    return logger


logger = get_logger()


def transition_to_production_best_model(
    client,
    _logger,
    staging_eval_metric,
    production_eval_metric,
    staging_model_version,
    model_name,
):
    if staging_eval_metric <= production_eval_metric:
        _logger.info(
            "Candidate Staging model DOES NOT perform better than current Production model"
        )
        _logger.info(
            'Transition candidate model from stage="staging" to stage="archived"'
        )
        client.transition_model_version_stage(
            name=model_name, version=staging_model_version, stage="archived"
        )

    elif staging_eval_metric > production_eval_metric:
        _logger.info(
            "Candidate Staging model DOES perform better than current Production model"
        )
        _logger.info(
            'Transition candidate model from stage="staging" to stage="production"'
        )
        _logger.info("Existing Production model will be archived")
        client.transition_model_version_stage(
            name=model_name,
            version=staging_model_version,
            stage="production",
            archive_existing_versions=True,
        )


# COMMAND ----------

# If the staging model has a better f1 score than the production one, we replace the model in production with the new one.
transition_to_production_best_model(
    client, logger, staging_score, prod_score, staging_model_version, fs_model_name
)

# COMMAND ----------
