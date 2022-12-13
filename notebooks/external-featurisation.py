# Databricks notebook source
# MAGIC %md
# MAGIC ### External Featurization
# MAGIC This notebook mocks the creation of features by another Data Scientist into the Feature Store.

# COMMAND ----------

# MAGIC %md
# MAGIC We use the data from `bureau.csv` which contains external credit applications, to extract numerical aggregations features.

# COMMAND ----------

import pyspark.pandas as ps

bureau = spark.read.table("hive_metastore.home_credit_risk.bronze_bureau").pandas_api()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creation of new features
# MAGIC Group by the `SK_ID_CURR`, calculate aggregation statistics for number of credits etc, and reformat columns names.

# COMMAND ----------

bureau_agg = (
    bureau.drop(columns=["SK_ID_BUREAU"])
    .groupby("SK_ID_CURR", as_index=False)
    .agg(["count", "mean", "max", "min", "sum"])
    .reset_index(drop=True)
)

new_columns_names = ["_".join(col) for col in bureau_agg.columns.values]
bureau_agg.columns = new_columns_names
bureau_agg.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creation of Feature Store
# MAGIC A feature store is a centralized repository that enables data scientists to find and share features and also ensures that the same code used to compute the feature values is used for model training and inference.

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

try:
    fs.create_table(
        name="feature_store_home_credit.external_credit_application",
        primary_keys=["SK_ID_CURR"],
        df=bureau_agg.to_spark(index_col="SK_ID_CURR"),
        description="Customer external credit applications features, such as number of credits, overdue days, credit current activity.",
    )
except ValueError as e:
    print(e)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Writing features to Feature Store

# COMMAND ----------

fs.write_table(
    name="feature_store_home_credit.external_credit_application",
    df=bureau_agg.to_spark(index_col="SK_ID_CURR"),
    mode="merge",
)
