# Databricks notebook source
# MAGIC %md
# MAGIC ### Evidently
# MAGIC Evidently is an open-source Python library for data scientists and ML engineers for monitoring. 
# MAGIC 
# MAGIC It helps to estimate and explore data drift for data quality and machine learning models. The library can be found here on [GitHub](https://github.com/evidentlyai/evidently).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup

# COMMAND ----------

# MAGIC %pip install evidently

# COMMAND ----------

import os
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_preset import DataQualityTestPreset
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import TargetDriftPreset

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Reports Folder
# MAGIC Create a folder to store html reports we will generate.

# COMMAND ----------

if not os.path.exists('../reports'):
    os.makedirs('../reports')


# COMMAND ----------

# MAGIC %md 
# MAGIC #### Load Data
# MAGIC We are now loading reference data (historical) and production data (new data) to compare them against each others.

# COMMAND ----------

reference_table_name: str = 'hive_metastore.home_credit_risk.bronze_home_credit'
reference_df = spark.table(reference_table_name).toPandas().sample(5000)


# COMMAND ----------

production_table_name: str = 'hive_metastore.home_credit_risk.bronze_new_home_credit'
production_df = spark.table(production_table_name).toPandas().sample(5000)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Run Data Quality Test

# COMMAND ----------

data_quality = TestSuite(tests=[DataQualityTestPreset()])
data_quality.run(current_data=production_df, reference_data=reference_df, column_mapping=None)
data_quality.save_html("../reports/data_quality.html")

# COMMAND ----------

# To show the reports, uncomment these lines
data_quality_html = open("../reports/data_quality.html", 'r').read()
displayHTML(data_quality_html)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run Data Stability Test

# COMMAND ----------

data_stability= TestSuite(tests=[DataStabilityTestPreset()])
data_stability.run(current_data=production_df, reference_data=reference_df, column_mapping=None)
data_stability.save_html("../reports/data_stability.html")

# COMMAND ----------

# To show the reports, uncomment these lines
data_stability_html = open("../reports/data_stability.html", 'r').read()
displayHTML(data_stability_html)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run Data Drift Test

# COMMAND ----------

data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(current_data=production_df, reference_data=reference_df, column_mapping=None)
data_drift_report.save_html("../reports/data_drift_report.html")

# COMMAND ----------

# To show the reports, uncomment these lines
data_drift_report_html = open("../reports/data_drift_report.html", 'r').read()
displayHTML(data_drift_report_html)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Run Model Monitoring Dashboard
# MAGIC The aim is to track the performance of our current model in production on historical data (reference) and new data coming in (production). 
# MAGIC 
# MAGIC If we notice a degradation of the model performance in our production dataset, we can decide to re-train the model.

# COMMAND ----------

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from databricks.feature_store import FeatureStoreClient
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import ClassificationPerformanceTab
from evidently.pipeline.column_mapping import ColumnMapping

# COMMAND ----------

fs = FeatureStoreClient()
client = MlflowClient() 

# Fetch the mlflow uri of the current model in production
model_name = 'fs_home_credit_model'
production_model_version = client.get_latest_versions(name=model_name, stages=['production'])[0].version
production_uri = f"models:/{model_name}/{production_model_version}"

# COMMAND ----------

# Predict for historical data
reference_spark_df = spark.createDataFrame(reference_df)
reference_spark_df = reference_spark_df.withColumn("SK_ID_CURR", reference_spark_df.SK_ID_CURR.cast('long'))
reference_model_predictions = fs.score_batch(production_uri, reference_spark_df, result_type='int
                                             
# Predict for new data
production_spark_df = spark.createDataFrame(production_df)
production_spark_df = production_spark_df.withColumn("SK_ID_CURR", production_spark_df.SK_ID_CURR.cast('long'))
production_model_predictions = fs.score_batch(production_uri, production_spark_df, result_type='int')

# COMMAND ----------

# Mapping dataset columns for evidently dashboard function
column_mapping = ColumnMapping()
column_mapping.target = "TARGET"
column_mapping.prediction = "prediction"
column_mapping.numerical_features = ["EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT", "AMT_INCOME_TOTAL", "APARTMENTS_AVG", "BASEMENTAREA_AVG", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "DAYS_BIRTH", "DAYS_EMPLOYED"]
column_mapping.categorical_features = ["ORGANIZATION_TYPE", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "CODE_GENDER", "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "EMERGENCYSTATE_MODE", "FLAG_CONT_MOBILE", "FLAG_DOCUMENT_10"]

# COMMAND ----------

classification_dashboard = Dashboard(tabs=[ClassificationPerformanceTab(verbose_level=1)])
classification_dashboard.calculate(reference_model_predictions.toPandas(), production_model_predictions.toPandas(), column_mapping = column_mapping)
classification_dashboard.save("../reports/classification_dashboard.html")

# COMMAND ----------

# To show the reports, uncomment these lines
classification_dashboard_html = open("../reports/classification_dashboard.html", 'r').read()
displayHTML(classification_dashboard_html)
