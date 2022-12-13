# Databricks notebook source
# MAGIC %md
# MAGIC ### Feature Selection
# MAGIC Feature selection is the process of reducing the number of input variables when developing a predictive model.
# MAGIC It is desirable to reduce the number of input variables to both reduce the computational cost of modeling and, in some cases, to improve the performance of the model.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup

# COMMAND ----------

pip install plotly

# COMMAND ----------

from pyspark import pandas as pd  # To use Spark with pandas syntax
import plotly.offline as py
import plotly.graph_objs as go

# COMMAND ----------

credit_table_path = "hive_metastore.home_credit_risk.bronze_home_credit"
df_credit_application = spark.read.table(credit_table_path).pandas_api()

# Preview data
df_credit_application.head(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Feature Store
# MAGIC Here, we are reading features from Feature Store Table `external_credit_application`
# MAGIC We want to check this existing Feature table and use these features for our model training.

# COMMAND ----------

# Import and create Feature Store client
from databricks import feature_store
fs = feature_store.FeatureStoreClient()

# Load features
external_credit_table_path = "feature_store_home_credit.external_credit_application"
external_credit_application_features = fs.read_table(external_credit_table_path).pandas_api()

# Preview features
external_credit_application_features.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we merge the training dataset with these existing features we found from the Feature Store.

# COMMAND ----------

merged_dataset = df_credit_application.merge(external_credit_application_features, on="SK_ID_CURR").to_pandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Correlation Plot

# COMMAND ----------

data = [
    go.Heatmap(
        z=merged_dataset.corr().values,
        x=merged_dataset.columns.values,
        y=merged_dataset.columns.values,
        colorscale='Viridis',
        reversescale = False,
        opacity = 1.0)
]

layout = go.Layout(
    title='Pearson Correlation of features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 700,
    margin=dict(l=240,),)

fig = go.Figure(data=data, layout=layout)
fig.show(renderer='databricks')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Importance
# MAGIC We are here extracting feature importance using a Random forest model

# COMMAND ----------

from sklearn import preprocessing

categorical_feats = [f for f in merged_dataset.columns if merged_dataset[f].dtype == 'object']
for col in categorical_feats:
    lb = preprocessing.LabelEncoder()
    lb.fit(list(merged_dataset[col].values.astype('str')))
    merged_dataset[col] = lb.transform(list(merged_dataset[col].values.astype('str')))

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

feature_importance_dataset = merged_dataset.fillna(-999)
rf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=2018)
rf.fit(feature_importance_dataset.drop(['SK_ID_CURR', 'TARGET'],axis=1), feature_importance_dataset.TARGET)
features = feature_importance_dataset.drop(['SK_ID_CURR', 'TARGET'],axis=1).columns.values

# COMMAND ----------

# MAGIC %md
# MAGIC Plotting the feature importances using Plotly

# COMMAND ----------

x, y = (list(x) for x in zip(*sorted(zip(rf.feature_importances_, features), reverse = False)))
feature_importance_figure = go.Bar(
    x=x, y=y, marker=dict(
                    color=x,
                    colorscale = 'Viridis',
                    reversescale = True
    ), name='Random Forest Feature importance', orientation='h')

layout = dict(
    title='Barplot of Feature importances',
    width = 900, height = 2000,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
    ), margin=dict(l=300))

fig1 = go.Figure(data=[feature_importance_figure])
fig1['layout'].update(layout)
fig1.show(renderer='databricks')

# COMMAND ----------

# MAGIC %md
# MAGIC Select the best 50 features for model training using AutoML.

# COMMAND ----------

best_scores, feature_names = (list(x) for x in zip(*sorted(zip(rf.feature_importances_, features), reverse = True)))
best_50_features = feature_names[:50]
best_50_features

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Table for AutoML
# MAGIC Saving the table with the 50 best features to train later in AutoML.

# COMMAND ----------

automl_df = merged_dataset[['SK_ID_CURR', 'TARGET'] + best_50_features]

# COMMAND ----------

database_name = 'home_credit_risk'
automl_path = '/FileStore/dataset/home-credit-default-risk/bronze/automl_features/' 
automl_tbl_name = 'automl_features'

# For saving dataset that will be used for AutoML
_ = pd.from_pandas(automl_df).to_delta(path=automl_path, mode='overwrite', mergeSchema=True)
_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name, automl_tbl_name, automl_path))
