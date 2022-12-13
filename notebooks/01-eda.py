# Databricks notebook source
# MAGIC %md
# MAGIC ### Exploratory Data Analysis

# COMMAND ----------

from pyspark import pandas as pd  # To use Spark with pandas syntax

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading Data
# MAGIC 
# MAGIC The data is provided by Home Credit, a service dedicated to provided lines of credit (loans) to the unbanked population. Predicting whether or not a client will repay a loan or have difficulty is a critical business need, and Home Credit is hosting this competition on Kaggle to see what sort of models the machine learning community can develop to help them in this task.
# MAGIC 
# MAGIC There are different sources of data:
# MAGIC 
# MAGIC - **application_train/application_test**: the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature SK_ID_CURR. The training application data comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not repaid.
# MAGIC - **bureau**: data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits.

# COMMAND ----------

# Set config for file path
data_path = '/FileStore/dataset/home-credit-default-risk' 

# COMMAND ----------

application_train_df = pd.read_csv(f"{data_path}/application_train.csv")
print('Training data shape: ', application_train_df.shape)
application_train_df.head()

# COMMAND ----------

# Testing data features
application_test_df = pd.read_csv(f"{data_path}/application_test.csv")
print('Testing data shape: ', application_test_df.shape)
application_test_df.head()

# COMMAND ----------

bureau_df = pd.read_csv(f"{data_path}/bureau.csv")
print('Bureau data shape: ', bureau_df.shape)
bureau_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Data Profiling

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data profiling with internal profiling

# COMMAND ----------

display(application_train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data profiling with pandas_profiling

# COMMAND ----------

from pandas_profiling import ProfileReport
df_profile = ProfileReport(application_test_df.to_pandas(), minimal=True, title="Profiling Report", progress_bar=True, infer_dtypes=False)
displayHTML(df_profile.to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC ##### With Pandas on Spark

# COMMAND ----------

organization_count_df = application_train_df["ORGANIZATION_TYPE"].value_counts()
organization_count_df.plot(kind='bar')

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### With Bamboolib

# COMMAND ----------

!pip install bamboolib

# COMMAND ----------

import bamboolib as bam
bam

# COMMAND ----------

# MAGIC %md
# MAGIC ##### With Plotly

# COMMAND ----------

import plotly.express as px
fig = px.histogram(df.sample(n=10000, replace=True, random_state=123).sort_index(), x='ORGANIZATION_TYPE', color='TARGET')
fig.update_xaxes(categoryorder='total descending')
fig

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlations
# MAGIC 
# MAGIC Now that we have dealt with the categorical variables and the outliers, let's continue with the EDA. One way to try and understand the data is by looking for correlations between the features and the target. We can calculate the Pearson correlation coefficient between every variable and the target using the .corr dataframe method.
# MAGIC 
# MAGIC The correlation coefficient is not the greatest method to represent "relevance" of a feature, but it does give us an idea of possible relationships within the data. Some general interpretations of the absolute value of the correlation coefficent are:
# MAGIC 
# MAGIC - .00-.19 “very weak”
# MAGIC - .20-.39 “weak”
# MAGIC - .40-.59 “moderate”
# MAGIC - .60-.79 “strong”
# MAGIC - .80-1.0 “very strong”

# COMMAND ----------

# Find correlations with the target and sort
correlations = application_train_df.to_pandas().corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))
