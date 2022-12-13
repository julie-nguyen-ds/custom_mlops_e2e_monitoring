# Databricks notebook source
import mlflow
import shutil
from pyspark import pandas as pd  # To use Spark with pandas syntax

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting up Environments & Datasets
# MAGIC This notebook helps set up the different component that we will use for in our use case for MLOps, features and experimentations, notably:
# MAGIC - Feature Store
# MAGIC - Mlflow Experiments
# MAGIC - Delta Lake format to Data Tables

# COMMAND ----------

# User
user = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
    .split("@")[0]
)
username_sql = user.replace(".", "_")

# Database
use_case_name = "home_credit_risk"
database_name = f"database_{use_case_name}"
data_path = f"/FileStore/dataset/{use_case_name}"
test_data_dbfs_filepath = f"{data_path}/application_test.csv"
bronze_tbl_path = f"{data_path}/bronze/"
bronze_tbl_name = f"bronze_{use_case_name}"
test_data_tbl_name = f"bronze_test_{use_case_name}"
upstream_tbl_name = f"bronze_new_{use_case_name}"
inference_data_tbl_name = f"inference_{use_case_name}"
bronze_bureau_tbl_name = "bronze_bureau"

# Feature Store
feature_store_database_name = f"feature_store_{use_case_name}"

# Mlflow Experiment path
mlflow_experiment_path = f"/Shared/Mlflow_{use_case_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Store Creation

# COMMAND ----------

# Create database to house feature tables
_ = spark.sql(f"CREATE DATABASE IF NOT EXISTS {feature_store_database_name}")
#fs.drop_table(
#                name=feature_store_table
#            )
# COMMAND ----------

# MAGIC %md
# MAGIC ### MLFlow Experiment Creation
# MAGIC The MLflow Tracking component is an API and UI
# MAGIC MLflow is an open source platform for managing machine learning workflows. It can be used for logging parameters, code versions, metrics, and output files when running your machine learning code and for later visualizing the results.
# MAGIC
# MAGIC An experiment is the basic unit of MLflow organization. All MLflow runs belong to an experiment. For each experiment, you can analyze and compare the results of different runs, and easily retrieve metadata artifacts for analysis using downstream tools. Experiments are maintained on an MLflow tracking server hosted on Azure Databricks.
# MAGIC
# MAGIC Here, we create an experiment to store future model training runs and keep track of them.

# COMMAND ----------

try:
    _ = mlflow.create_experiment(name=mlflow_experiment_path)
except Exception as e:
    print("Experiment already exists.", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Database Creation and Data Ingestion
# MAGIC Loading Raw data into delta lake format and to table.

# COMMAND ----------

# Delete the old database and tables if needed
_ = spark.sql(f"DROP DATABASE IF EXISTS {database_name} CASCADE")
shutil.rmtree("/dbfs" + bronze_tbl_path, ignore_errors=True)

# COMMAND ----------

# Create database to house tables
_ = spark.sql(f"CREATE DATABASE {database_name}")
_ = spark.sql(f"USE {database_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading csv files to Spark DataFrames
# MAGIC Read and store credit application csv files for featurization, model training, drift detection, inference.
# MAGIC
# MAGIC We split the main training data file `application_train.csv` into 3 parts for:
# MAGIC - Training the model (70%)
# MAGIC - New data for testing drift (20%)
# MAGIC - Model testing and comparison in production (10%)

# COMMAND ----------

application_df = pd.read_csv(f"{data_path}/application_train.csv")
print("Overall data shape: ", application_df.shape)

# Data for Training the model (70%)
application_train_df = application_df.sample(frac=0.7)
application_rest_df = application_df.drop(application_train_df.index.values)
print("Training data shape: ", application_train_df.shape)

# New data for testing drift (20%)
application_new_df = application_rest_df.sample(frac=0.66)
print("New data shape: ", application_new_df.shape)

# Data for Model testing and comparison in production environment (10%)
application_test_df = application_rest_df.drop(application_new_df.index.values)
print("Test data shape: ", application_test_df.shape)

application_inference_df = pd.read_csv(f"{data_path}/application_test.csv")
print("Inference data shape: ", application_inference_df.shape)

bureau_df = pd.read_csv(f"{data_path}/bureau.csv")
print("Bureau data shape: ", bureau_df.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the spark dataframe to delta lake formats.

# COMMAND ----------

_ = application_train_df.to_delta(
    path=bronze_tbl_path + f"{bronze_tbl_name}/", mode="overwrite", mergeSchema=True
)
_ = application_new_df.to_delta(
    path=bronze_tbl_path + f"{upstream_tbl_name}/", mode="overwrite", mergeSchema=True
)
_ = application_test_df.to_delta(
    path=bronze_tbl_path + f"{test_data_tbl_name}/", mode="overwrite", mergeSchema=True
)
_ = application_inference_df.to_delta(
    path=bronze_tbl_path + f"{inference_data_tbl_name}/",
    mode="overwrite",
    mergeSchema=True,
)
_ = bureau_df.to_delta(
    path=bronze_tbl_path + f"{bronze_bureau_tbl_name}/",
    mode="overwrite",
    mergeSchema=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Create tables from the delta lake formats using Spark SQL. The data is now ready to use.

# COMMAND ----------

# For training application data
_ = spark.sql(
    """
  CREATE TABLE IF NOT EXISTS `{}`.{} 
  USING DELTA 
  LOCATION '{}'
  """.format(
        database_name, bronze_tbl_name, bronze_tbl_path + "/" + bronze_tbl_name
    )
)

# For new application data
_ = spark.sql(
    """
  CREATE TABLE IF NOT EXISTS `{}`.{} 
  USING DELTA 
  LOCATION '{}'
  """.format(
        database_name, upstream_tbl_name, bronze_tbl_path + "/" + upstream_tbl_name
    )
)

# For test application data
_ = spark.sql(
    """
  CREATE TABLE IF NOT EXISTS `{}`.{} 
  USING DELTA 
  LOCATION '{}'
  """.format(
        database_name, test_data_tbl_name, bronze_tbl_path + "/" + test_data_tbl_name
    )
)

# For inference application data
_ = spark.sql(
    """
  CREATE TABLE IF NOT EXISTS `{}`.{} 
  USING DELTA 
  LOCATION '{}'
  """.format(
        database_name,
        inference_data_tbl_name,
        bronze_tbl_path + "/" + inference_data_tbl_name,
    )
)

# For bureau data
_ = spark.sql(
    """
  CREATE TABLE IF NOT EXISTS `{}`.{}
  USING DELTA 
  LOCATION '{}'
  """.format(
        database_name,
        bronze_bureau_tbl_name,
        bronze_tbl_path + "/" + bronze_bureau_tbl_name,
    )
)

# COMMAND ----------
