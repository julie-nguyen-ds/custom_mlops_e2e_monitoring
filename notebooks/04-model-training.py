# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training
# MAGIC This is an auto-generated notebook. To reproduce these results, attach this notebook to the **julie.nguyen@databricks.com's Cluster** cluster and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/1968295001928865)
# MAGIC - Navigate to the parent notebook [here](#notebook/1968295001928866) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.
# MAGIC
# MAGIC Runtime Version: _11.3.x-cpu-ml-scala2.12_

# COMMAND ----------

import mlflow
import databricks.automl_runtime
from sklearn.model_selection import train_test_split

target_col = "TARGET"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

from mlflow.tracking import MlflowClient

input_client = MlflowClient()

import os
import pandas as pd

df_loaded = spark.read.table("hive_metastore.home_credit_risk.bronze_home_credit")
df_loaded = df_loaded.withColumn("SK_ID_CURR", df_loaded.SK_ID_CURR.cast("long"))

# Preview data
display(df_loaded.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector

supported_cols = [
    "YEARS_BUILD_MODE",
    "FLOORSMAX_MODE",
    "LIVINGAPARTMENTS_MEDI",
    "FLAG_DOCUMENT_3",
    "FLAG_OWN_REALTY",
    "FLOORSMIN_MODE",
    "WEEKDAY_APPR_PROCESS_START",
    "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "NONLIVINGAPARTMENTS_MEDI",
    "DAYS_LAST_PHONE_CHANGE",
    "REG_REGION_NOT_WORK_REGION",
    "FLAG_DOCUMENT_15",
    "FLAG_DOCUMENT_7",
    "AMT_INCOME_TOTAL",
    "FLAG_MOBIL",
    "HOUR_APPR_PROCESS_START",
    "NONLIVINGAPARTMENTS_AVG",
    "LIVINGAPARTMENTS_AVG",
    "REGION_RATING_CLIENT_W_CITY",
    "APARTMENTS_MODE",
    "NAME_INCOME_TYPE",
    "ELEVATORS_MEDI",
    "FLOORSMAX_MEDI",
    "EXT_SOURCE_1",
    "BASEMENTAREA_AVG",
    "APARTMENTS_MEDI",
    "FLAG_DOCUMENT_9",
    "AMT_ANNUITY",
    "APARTMENTS_AVG",
    "FLAG_WORK_PHONE",
    "FONDKAPREMONT_MODE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "ENTRANCES_AVG",
    "YEARS_BEGINEXPLUATATION_MODE",
    "NONLIVINGAPARTMENTS_MODE",
    "WALLSMATERIAL_MODE",
    "LIVE_REGION_NOT_WORK_REGION",
    "LIVINGAPARTMENTS_MODE",
    "NAME_HOUSING_TYPE",
    "FLAG_DOCUMENT_21",
    "LANDAREA_MEDI",
    "REG_REGION_NOT_LIVE_REGION",
    "OCCUPATION_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_DOCUMENT_14",
    "FLOORSMIN_MEDI",
    "COMMONAREA_AVG",
    "NONLIVINGAREA_MEDI",
    "ENTRANCES_MODE",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "COMMONAREA_MEDI",
    "LIVE_CITY_NOT_WORK_CITY",
    "TOTALAREA_MODE",
    "FLAG_CONT_MOBILE",
    "FLAG_EMP_PHONE",
    "FLAG_DOCUMENT_8",
    "ELEVATORS_MODE",
    "NONLIVINGAREA_AVG",
    "FLAG_DOCUMENT_10",
    "CNT_CHILDREN",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "CNT_FAM_MEMBERS",
    "FLAG_DOCUMENT_4",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "FLAG_PHONE",
    "FLAG_DOCUMENT_18",
    "EXT_SOURCE_2",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "COMMONAREA_MODE",
    "DAYS_BIRTH",
    "NONLIVINGAREA_MODE",
    "ORGANIZATION_TYPE",
    "YEARS_BUILD_AVG",
    "REGION_POPULATION_RELATIVE",
    "BASEMENTAREA_MODE",
    "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_5",
    "DAYS_EMPLOYED",
    "NAME_CONTRACT_TYPE",
    "YEARS_BEGINEXPLUATATION_MEDI",
    "OWN_CAR_AGE",
    "EMERGENCYSTATE_MODE",
    "LANDAREA_MODE",
    "YEARS_BEGINEXPLUATATION_AVG",
    "FLAG_DOCUMENT_2",
    "REGION_RATING_CLIENT",
    "FLAG_DOCUMENT_12",
    "FLAG_DOCUMENT_20",
    "LANDAREA_AVG",
    "YEARS_BUILD_MEDI",
    "REG_CITY_NOT_LIVE_CITY",
    "AMT_CREDIT",
    "LIVINGAREA_MEDI",
    "FLAG_DOCUMENT_16",
    "FLAG_EMAIL",
    "REG_CITY_NOT_WORK_CITY",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "LIVINGAREA_AVG",
    "BASEMENTAREA_MEDI",
    "NAME_FAMILY_STATUS",
    "FLOORSMIN_AVG",
    "FLAG_DOCUMENT_11",
    "FLOORSMAX_AVG",
    "NAME_TYPE_SUITE",
    "DAYS_ID_PUBLISH",
    "NAME_EDUCATION_TYPE",
    "FLAG_DOCUMENT_19",
    "AMT_GOODS_PRICE",
    "FLAG_DOCUMENT_6",
    "DAYS_REGISTRATION",
    "LIVINGAREA_MODE",
    "ENTRANCES_MEDI",
    "CODE_GENDER",
    "FLAG_DOCUMENT_17",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "HOUSETYPE_MODE",
    "SK_ID_CURR",
    "ELEVATORS_AVG",
]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(
    (
        "impute_mean",
        SimpleImputer(),
        [
            "AMT_ANNUITY",
            "AMT_CREDIT",
            "AMT_GOODS_PRICE",
            "AMT_INCOME_TOTAL",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
            "APARTMENTS_AVG",
            "APARTMENTS_MEDI",
            "APARTMENTS_MODE",
            "BASEMENTAREA_AVG",
            "BASEMENTAREA_MEDI",
            "BASEMENTAREA_MODE",
            "CNT_CHILDREN",
            "CNT_FAM_MEMBERS",
            "COMMONAREA_AVG",
            "COMMONAREA_MEDI",
            "COMMONAREA_MODE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "DAYS_ID_PUBLISH",
            "DAYS_LAST_PHONE_CHANGE",
            "DAYS_REGISTRATION",
            "DEF_30_CNT_SOCIAL_CIRCLE",
            "DEF_60_CNT_SOCIAL_CIRCLE",
            "ELEVATORS_AVG",
            "ELEVATORS_MEDI",
            "ELEVATORS_MODE",
            "ENTRANCES_AVG",
            "ENTRANCES_MEDI",
            "ENTRANCES_MODE",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "FLAG_CONT_MOBILE",
            "FLAG_DOCUMENT_10",
            "FLAG_DOCUMENT_11",
            "FLAG_DOCUMENT_12",
            "FLAG_DOCUMENT_13",
            "FLAG_DOCUMENT_14",
            "FLAG_DOCUMENT_15",
            "FLAG_DOCUMENT_16",
            "FLAG_DOCUMENT_17",
            "FLAG_DOCUMENT_18",
            "FLAG_DOCUMENT_19",
            "FLAG_DOCUMENT_2",
            "FLAG_DOCUMENT_20",
            "FLAG_DOCUMENT_21",
            "FLAG_DOCUMENT_3",
            "FLAG_DOCUMENT_4",
            "FLAG_DOCUMENT_5",
            "FLAG_DOCUMENT_6",
            "FLAG_DOCUMENT_7",
            "FLAG_DOCUMENT_8",
            "FLAG_DOCUMENT_9",
            "FLAG_EMAIL",
            "FLAG_EMP_PHONE",
            "FLAG_MOBIL",
            "FLAG_PHONE",
            "FLAG_WORK_PHONE",
            "FLOORSMAX_AVG",
            "FLOORSMAX_MEDI",
            "FLOORSMAX_MODE",
            "FLOORSMIN_AVG",
            "FLOORSMIN_MEDI",
            "FLOORSMIN_MODE",
            "HOUR_APPR_PROCESS_START",
            "LANDAREA_AVG",
            "LANDAREA_MEDI",
            "LANDAREA_MODE",
            "LIVE_CITY_NOT_WORK_CITY",
            "LIVE_REGION_NOT_WORK_REGION",
            "LIVINGAPARTMENTS_AVG",
            "LIVINGAPARTMENTS_MEDI",
            "LIVINGAPARTMENTS_MODE",
            "LIVINGAREA_AVG",
            "LIVINGAREA_MEDI",
            "LIVINGAREA_MODE",
            "NONLIVINGAPARTMENTS_AVG",
            "NONLIVINGAPARTMENTS_MEDI",
            "NONLIVINGAPARTMENTS_MODE",
            "NONLIVINGAREA_AVG",
            "NONLIVINGAREA_MEDI",
            "NONLIVINGAREA_MODE",
            "OBS_30_CNT_SOCIAL_CIRCLE",
            "OBS_60_CNT_SOCIAL_CIRCLE",
            "OWN_CAR_AGE",
            "REGION_POPULATION_RELATIVE",
            "REGION_RATING_CLIENT",
            "REGION_RATING_CLIENT_W_CITY",
            "REG_CITY_NOT_LIVE_CITY",
            "REG_CITY_NOT_WORK_CITY",
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "SK_ID_CURR",
            "TOTALAREA_MODE",
            "YEARS_BEGINEXPLUATATION_AVG",
            "YEARS_BEGINEXPLUATATION_MEDI",
            "YEARS_BEGINEXPLUATATION_MODE",
            "YEARS_BUILD_AVG",
            "YEARS_BUILD_MEDI",
            "YEARS_BUILD_MODE",
        ],
    )
)

numerical_pipeline = Pipeline(
    steps=[
        (
            "converter",
            FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce")),
        ),
        ("imputers", ColumnTransformer(num_imputers)),
        ("standardizer", StandardScaler()),
    ]
)

numerical_transformers = [
    (
        "numerical",
        numerical_pipeline,
        [
            "LIVINGAPARTMENTS_MEDI",
            "YEARS_BUILD_MODE",
            "FLOORSMAX_MODE",
            "FLAG_DOCUMENT_3",
            "FLOORSMIN_MODE",
            "EXT_SOURCE_3",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "NONLIVINGAPARTMENTS_MEDI",
            "DAYS_LAST_PHONE_CHANGE",
            "REG_REGION_NOT_WORK_REGION",
            "FLAG_DOCUMENT_15",
            "FLAG_DOCUMENT_7",
            "AMT_INCOME_TOTAL",
            "FLAG_MOBIL",
            "HOUR_APPR_PROCESS_START",
            "NONLIVINGAPARTMENTS_AVG",
            "LIVINGAPARTMENTS_AVG",
            "REGION_RATING_CLIENT_W_CITY",
            "APARTMENTS_MODE",
            "ELEVATORS_MEDI",
            "FLOORSMAX_MEDI",
            "EXT_SOURCE_1",
            "BASEMENTAREA_AVG",
            "APARTMENTS_MEDI",
            "FLAG_DOCUMENT_9",
            "AMT_ANNUITY",
            "APARTMENTS_AVG",
            "FLAG_WORK_PHONE",
            "OBS_60_CNT_SOCIAL_CIRCLE",
            "ENTRANCES_AVG",
            "YEARS_BEGINEXPLUATATION_MODE",
            "NONLIVINGAPARTMENTS_MODE",
            "LIVE_REGION_NOT_WORK_REGION",
            "LIVINGAPARTMENTS_MODE",
            "FLAG_DOCUMENT_21",
            "LANDAREA_MEDI",
            "REG_REGION_NOT_LIVE_REGION",
            "FLAG_DOCUMENT_14",
            "FLOORSMIN_MEDI",
            "COMMONAREA_AVG",
            "NONLIVINGAREA_MEDI",
            "ENTRANCES_MODE",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
            "COMMONAREA_MEDI",
            "LIVE_CITY_NOT_WORK_CITY",
            "TOTALAREA_MODE",
            "FLAG_CONT_MOBILE",
            "NONLIVINGAREA_AVG",
            "FLAG_DOCUMENT_10",
            "ELEVATORS_MODE",
            "FLAG_DOCUMENT_8",
            "FLAG_EMP_PHONE",
            "CNT_CHILDREN",
            "DEF_30_CNT_SOCIAL_CIRCLE",
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "CNT_FAM_MEMBERS",
            "FLAG_DOCUMENT_4",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "OBS_30_CNT_SOCIAL_CIRCLE",
            "FLAG_PHONE",
            "FLAG_DOCUMENT_18",
            "EXT_SOURCE_2",
            "DEF_60_CNT_SOCIAL_CIRCLE",
            "COMMONAREA_MODE",
            "DAYS_BIRTH",
            "NONLIVINGAREA_MODE",
            "YEARS_BUILD_AVG",
            "REGION_POPULATION_RELATIVE",
            "BASEMENTAREA_MODE",
            "FLAG_DOCUMENT_13",
            "FLAG_DOCUMENT_5",
            "DAYS_EMPLOYED",
            "YEARS_BEGINEXPLUATATION_MEDI",
            "OWN_CAR_AGE",
            "LANDAREA_MODE",
            "YEARS_BEGINEXPLUATATION_AVG",
            "FLAG_DOCUMENT_2",
            "REGION_RATING_CLIENT",
            "FLAG_DOCUMENT_12",
            "FLAG_DOCUMENT_20",
            "LANDAREA_AVG",
            "YEARS_BUILD_MEDI",
            "REG_CITY_NOT_LIVE_CITY",
            "AMT_CREDIT",
            "LIVINGAREA_MEDI",
            "FLAG_DOCUMENT_16",
            "FLAG_EMAIL",
            "REG_CITY_NOT_WORK_CITY",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "LIVINGAREA_AVG",
            "BASEMENTAREA_MEDI",
            "FLOORSMIN_AVG",
            "FLAG_DOCUMENT_11",
            "FLOORSMAX_AVG",
            "DAYS_ID_PUBLISH",
            "FLAG_DOCUMENT_19",
            "AMT_GOODS_PRICE",
            "FLAG_DOCUMENT_6",
            "DAYS_REGISTRATION",
            "LIVINGAREA_MODE",
            "ENTRANCES_MEDI",
            "FLAG_DOCUMENT_17",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "SK_ID_CURR",
            "ELEVATORS_AVG",
        ],
    )
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_imputers = []

one_hot_pipeline = Pipeline(
    steps=[
        ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

categorical_one_hot_transformers = [
    (
        "onehot",
        one_hot_pipeline,
        [
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "CNT_CHILDREN",
            "CNT_FAM_MEMBERS",
            "CODE_GENDER",
            "DEF_30_CNT_SOCIAL_CIRCLE",
            "DEF_60_CNT_SOCIAL_CIRCLE",
            "EMERGENCYSTATE_MODE",
            "FLAG_CONT_MOBILE",
            "FLAG_DOCUMENT_10",
            "FLAG_DOCUMENT_11",
            "FLAG_DOCUMENT_12",
            "FLAG_DOCUMENT_13",
            "FLAG_DOCUMENT_14",
            "FLAG_DOCUMENT_15",
            "FLAG_DOCUMENT_16",
            "FLAG_DOCUMENT_17",
            "FLAG_DOCUMENT_18",
            "FLAG_DOCUMENT_19",
            "FLAG_DOCUMENT_2",
            "FLAG_DOCUMENT_20",
            "FLAG_DOCUMENT_21",
            "FLAG_DOCUMENT_3",
            "FLAG_DOCUMENT_4",
            "FLAG_DOCUMENT_5",
            "FLAG_DOCUMENT_6",
            "FLAG_DOCUMENT_7",
            "FLAG_DOCUMENT_8",
            "FLAG_DOCUMENT_9",
            "FLAG_EMAIL",
            "FLAG_EMP_PHONE",
            "FLAG_MOBIL",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "FLAG_PHONE",
            "FLAG_WORK_PHONE",
            "FONDKAPREMONT_MODE",
            "HOUSETYPE_MODE",
            "LIVE_CITY_NOT_WORK_CITY",
            "LIVE_REGION_NOT_WORK_REGION",
            "NAME_CONTRACT_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "NAME_INCOME_TYPE",
            "NAME_TYPE_SUITE",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE",
            "REGION_RATING_CLIENT",
            "REGION_RATING_CLIENT_W_CITY",
            "REG_CITY_NOT_LIVE_CITY",
            "REG_CITY_NOT_WORK_CITY",
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "WALLSMATERIAL_MODE",
            "WEEKDAY_APPR_PROCESS_START",
        ],
    )
]

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = numerical_transformers + categorical_one_hot_transformers
preprocessor = ColumnTransformer(
    transformers, remainder="passthrough", sparse_threshold=0
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC The input data is split by AutoML into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

from databricks.feature_store import FeatureLookup, FeatureStoreClient


# The model training uses two features from the 'customer_features' feature table and
# a single feature from 'product_features'
feature_lookups = [
    FeatureLookup(
        table_name="feature_store_home_credit.external_credit_application",
        feature_names=["CREDIT_ACTIVE_count", "CREDIT_DAY_OVERDUE_count"],
        lookup_key="SK_ID_CURR",
    )
]

fs = FeatureStoreClient()

# Create a training set using training DataFrame and features from Feature Store
# The training DataFrame must contain all lookup keys from the set of feature lookups,
# in this case 'customer_id' and 'product_id'. It must also contain all labels used
# for training, in this case 'rating'.
training_set = fs.create_training_set(
    df=df_loaded.sample(fraction=0.5), feature_lookups=feature_lookups, label="TARGET"
)

df_loaded_sample = training_set.load_df()
df_loaded_sample_pd = df_loaded_sample.toPandas()

# COMMAND ----------

# Split dataset train - validation - test split internally
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df_loaded_sample_pd.drop([target_col], axis=1),
    df_loaded_sample_pd[target_col],
    test_size=0.4,
    random_state=1,
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.5, random_state=1
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/1968295001928865)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the objective function
# MAGIC The objective function used to find optimal hyperparameters. By default, this notebook only runs
# MAGIC this function once (`max_evals=1` in the `hyperopt.fmin` invocation) with fixed hyperparameters, but
# MAGIC hyperparameters can be tuned by modifying `space`, defined below. `hyperopt.fmin` will then use this
# MAGIC function's return value to search the space to minimize the loss.

# COMMAND ----------

from pathlib import Path

Path.cwd().joinpath("mlruns").as_uri()

# COMMAND ----------

import mlflow

experiment_name = "/Shared/Mlflow_home_credit_risk"
current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
experiment_id = current_experiment["experiment_id"]
fs_model_name = "fs_home_credit_model"

# COMMAND ----------

import lightgbm
from lightgbm import LGBMClassifier


# COMMAND ----------

from sklearn import set_config
from sklearn.pipeline import Pipeline
import mlflow

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
pipeline_val = Pipeline(
    [
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
    ]
)
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)

from hyperopt import hp, tpe, fmin, STATUS_OK, Trials


def objective(params):
    with mlflow.start_run(experiment_id=experiment_id, run_name="lgbm") as mlflow_run:
        skdtc_classifier = LGBMClassifier(**params)

        model = Pipeline(
            [
                ("column_selector", col_selector),
                ("preprocessor", preprocessor),
                ("classifier", skdtc_classifier),
            ]
        )

        model.fit(
            X_train,
            y_train,
            classifier__callbacks=[
                lightgbm.early_stopping(20),
                lightgbm.log_evaluation(0),
            ],
            classifier__eval_set=[(X_val_processed, y_val)],
        )

        # Log metrics for the training set
        skdtc_training_metrics = mlflow.sklearn.eval_and_log_metrics(
            model, X_train, y_train, prefix="training_", pos_label=1
        )

        # Log metrics for the validation set
        skdtc_val_metrics = mlflow.sklearn.eval_and_log_metrics(
            model, X_val, y_val, prefix="val_", pos_label=1
        )

        # Log metrics for the test set
        skdtc_test_metrics = mlflow.sklearn.eval_and_log_metrics(
            model, X_test, y_test, prefix="test_", pos_label=1
        )

        loss = skdtc_val_metrics["val_f1_score"]

        # Truncate metric key names so they can be displayed together
        skdtc_val_metrics = {
            k.replace("val_", ""): v for k, v in skdtc_val_metrics.items()
        }
        skdtc_test_metrics = {
            k.replace("test_", ""): v for k, v in skdtc_test_metrics.items()
        }

        fs.log_model(
            model,
            fs_model_name,
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=fs_model_name,
        )

    return {
        "loss": loss,
        "status": STATUS_OK,
        "val_metrics": skdtc_val_metrics,
        "test_metrics": skdtc_test_metrics,
        "model": model,
        "run": mlflow_run,
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure the hyperparameter search space
# MAGIC Configure the search space of parameters. Parameters below are all constant expressions but can be
# MAGIC modified to widen the search space. For example, when training a decision tree classifier, to allow
# MAGIC the maximum tree depth to be either 2 or 3, set the key of 'max_depth' to
# MAGIC `hp.choice('max_depth', [2, 3])`. Be sure to also increase `max_evals` in the `fmin` call below.
# MAGIC
# MAGIC See https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html
# MAGIC for more information on hyperparameter tuning as well as
# MAGIC http://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for documentation on supported
# MAGIC search expressions.
# MAGIC
# MAGIC For documentation on parameters used by the model in use, please see:
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# MAGIC
# MAGIC NOTE: The above URL points to a stable version of the documentation corresponding to the last
# MAGIC released version of the package. The documentation may differ slightly for the package version
# MAGIC used by this notebook.

# COMMAND ----------

space = {
    "colsample_bytree": 0.6265174568825117,
    "lambda_l1": 7.965729088621257,
    "lambda_l2": 4.493164626667399,
    "learning_rate": 1.6617539184863863,
    "max_bin": 135,
    "max_depth": 7,
    "min_child_samples": 164,
    "n_estimators": 76,
    "num_leaves": 5,
    "path_smooth": 2.1127991062641183,
    "subsample": 0.7816765110373729,
    "random_state": 198631309,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run trials
# MAGIC When widening the search space and training multiple models, switch to `SparkTrials` to parallelize
# MAGIC training on Spark:
# MAGIC ```
# MAGIC from hyperopt import SparkTrials
# MAGIC trials = SparkTrials()
# MAGIC ```
# MAGIC
# MAGIC NOTE: While `Trials` starts an MLFlow run for each set of hyperparameters, `SparkTrials` only starts
# MAGIC one top-level run; it will start a subrun for each set of hyperparameters.
# MAGIC
# MAGIC See http://hyperopt.github.io/hyperopt/scaleout/spark/ for more info.

# COMMAND ----------

trials = Trials()
fmin(
    objective,
    space=space,
    algo=tpe.suggest,
    max_evals=1,  # Increase this when widening the hyperparameter search space.
    trials=trials,
)

best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
    pd.DataFrame(
        [best_result["val_metrics"], best_result["test_metrics"]],
        index=["validation", "test"],
    )
)

set_config(display="diagram")
model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transitioning model to Staging

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

mlflow_client = MlflowClient()

latest_model_version = mlflow_client.get_latest_versions(fs_model_name)[0].version
mlflow_client.transition_model_version_stage(
    name=fs_model_name, version=latest_model_version, stage="Staging"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC
# MAGIC model_uri=f"models:/{model_name}/{model_version}"
# MAGIC model = mlflow.pyfunc.load_model(model_uri=model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC
# MAGIC model = mlflow.pyfunc.load_model(model_uri=model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix, ROC and Precision-Recall curves for validation data
# MAGIC
# MAGIC We show the confusion matrix, ROC and Precision-Recall curves of the model on the validation data.
# MAGIC
# MAGIC For the plots evaluated on the training and the test data, check the artifacts on the MLflow run page.

# COMMAND ----------

import uuid
from IPython.display import Image

# Create temp directory to download MLflow model artifact
eval_temp_dir = os.path.join(
    os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8]
)
os.makedirs(eval_temp_dir, exist_ok=True)

# Download the artifact
eval_path = mlflow.artifacts.download_artifacts(
    run_id=mlflow_run.info.run_id, dst_path=eval_temp_dir
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion matrix for validation dataset

# COMMAND ----------

eval_confusion_matrix_path = os.path.join(eval_path, "val_confusion_matrix.png")
display(Image(filename=eval_confusion_matrix_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROC curve for validation dataset

# COMMAND ----------

eval_roc_curve_path = os.path.join(eval_path, "val_roc_curve.png")
display(Image(filename=eval_roc_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision-Recall curve for validation dataset

# COMMAND ----------

eval_pr_curve_path = os.path.join(eval_path, "val_precision_recall_curve.png")
display(Image(filename=eval_pr_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpretability
# MAGIC
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = False

if shap_enabled:
    from shap import KernelExplainer, summary_plot

    # SHAP cannot explain models using data with nulls.
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).
    mode = X_train.mode().iloc[0]

    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(
        n=min(100, X_train.shape[0]), random_state=198631309
    ).fillna(mode)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(100, X_val.shape[0]), random_state=198631309).fillna(
        mode
    )

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model.classes_)
