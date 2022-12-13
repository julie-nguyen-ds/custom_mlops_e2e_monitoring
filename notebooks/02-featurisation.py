# Databricks notebook source
from pyspark.sql.functions import col

# COMMAND ----------

# MAGIC %md
# MAGIC ### Featurisation
# MAGIC
# MAGIC In this notebook, we will explore making features by hand for the Home Credit Default Risk competition. In an earlier notebook, we used only the existing features in the feature store to train our model in AutoML.

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

external_credit_application_features = fs.read_table(
    "feature_store_home_credit.external_credit_application"
)
display(external_credit_application_features)

# COMMAND ----------

# MAGIC %md
# MAGIC For this feature engineering notebook, we will be preparing the following features:
# MAGIC
# MAGIC    1. The number of previous loans for every client
# MAGIC    2. The number of currently active / finished loans for every client
# MAGIC    3. The number of different types of loans for every client

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1. Number of previous loans for every client

# COMMAND ----------

bureau = spark.read.table("hive_metastore.home_credit_risk.bronze_bureau").pandas_api()

# COMMAND ----------

nb_previous_credit = (
    bureau.groupby("SK_ID_CURR", as_index=False)["SK_ID_BUREAU"]
    .count()
    .rename(columns={"SK_ID_BUREAU": "NB_PREVIOUS_CREDIT"})
)
nb_previous_credit.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2. Number of currently active / finished loans for every client

# COMMAND ----------

nb_credit_status = (
    bureau.groupby(["SK_ID_CURR", "CREDIT_ACTIVE"], as_index=False)
    .size()
    .unstack()
    .fillna(0)
)
flattened_nb_credit_status_columns = [
    ("NB_" + nb_credit_status.columns.name + "_" + col).replace(" ", "_")
    for col in nb_credit_status.columns.values
]
nb_credit_status.columns = flattened_nb_credit_status_columns
nb_credit_status.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3. Number of different types of loans for every client

# COMMAND ----------

nb_credit_types = (
    bureau.groupby(["SK_ID_CURR", "CREDIT_TYPE"], as_index=False)
    .size()
    .unstack()
    .fillna(0)
)
flattened_nb_credit_types_columns = [
    ("NB_" + nb_credit_types.columns.name + "_" + col).replace(" ", "_")
    for col in nb_credit_types.columns.values
]
nb_credit_types.columns = flattened_nb_credit_types_columns
nb_credit_types.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Merging features
# MAGIC Now, we can merge all the features we created into a single dataframe.

# COMMAND ----------

merged_count_features = nb_previous_credit.merge(
    nb_credit_status.reset_index(), on="SK_ID_CURR"
).merge(nb_credit_types.reset_index(), on="SK_ID_CURR")
merged_count_features.columns = [
    column.replace("(", "").replace(")", "").replace("-", "_")
    for column in merged_count_features.columns
]
merged_count_features.head()

# COMMAND ----------

merged_count_features = (
    merged_count_features.set_index("SK_ID_CURR")
    .to_spark(index_col="SK_ID_CURR")
    .withColumn("SK_ID_CURR", col("SK_ID_CURR").cast("long"))
)
merged_count_features.index = "SK_ID_CURR"

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Add features to existing table
# MAGIC We can now add the new features to the existing feature table on client external credit applications by using the function `write_table` with `mode=merge`.
# MAGIC
# MAGIC This will add our new features columns to the existing features dataset, and merge them using the primary key of the table, which is the current client's ID `SK_ID_CURR`.

# COMMAND ----------

fs.write_table(
    name="feature_store_home_credit.external_credit_application",
    df=merged_count_features,
    mode="merge",
)
