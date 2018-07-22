# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 11:40:59 2018
Perform a simple estimation using Gradient Boosting Machine
@author: Nathan
"""

import numpy as np
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator
# Type "conda install -c anaconda h2o" in Anaconda Command Prompt
import h2o
import os

os.getcwd()

h2o.init()
# Load training data
# app_train = h2o.import_file(path = '../Data/app_train_poly.csv')
app_train = pd.read_csv("../Data/app_train_with_bureau.csv")
app_train[["TARGET"]] = app_train[["TARGET"]].astype("object")

# Make sure that the H2O package reads the data as the correct types for
# each column
data_type_list = []
for col in app_train:
    data_type_list.append(app_train[col].dtype.name)
data_type_list = ["int" if x == "int64" else x for x in data_type_list]
data_type_list = ["float" if x == "float64" else x for x in data_type_list]
data_type_list = ["factor" if x == "object" else x for x in data_type_list]
app_train = h2o.H2OFrame(app_train, column_types = data_type_list)
print(pd.DataFrame(app_train.columns).to_string())

# Ensure to make the y variable a factor, otherwise the training result may
# return a negative number
app_train[["TARGET"]] = app_train[["TARGET"]].asfactor()
app_train[['EXT_SOURCE_1']] = app_train[['EXT_SOURCE_1']].asnumeric()

# Specify the model
gbm_model = H2OGradientBoostingEstimator()
# Train the model, put all parameters other than "TARGET" in, the first column is
# "TARGET", the y variable
gbm_model.train(x=app_train.names[2:], y = app_train.names[1], training_frame = app_train)
# Load test set
#app_test = h2o.import_file(path = '../Data/app_test_poly.csv')
app_test = pd.read_csv("../Data/app_test_with_bureau.csv")
data_type_list = []
# Make sure that the H2O package reads the data as the correct types for
# each column
for col in app_test:
    data_type_list.append(app_test[col].dtype.name)
data_type_list = ["int" if x == "int64" else x for x in data_type_list]
data_type_list = ["float" if x == "float64" else x for x in data_type_list]
data_type_list = ["factor" if x == "object" else x for x in data_type_list]
app_test = h2o.H2OFrame(app_test, column_types = data_type_list)
#app_test[['APARTMENTS_AVG']] = app_test[['APARTMENTS_AVG']].asnumeric()


results = gbm_model.predict(test_data = app_test)
print(pd.DataFrame(app_test.columns).to_string())
# View the model if you want
gbm_model.show()
print(results.describe())

# Save the model
model_path = h2o.save_model(gbm_model, path= os.getcwd(), force=True)

# Use this if you gave your model a custom name
# model_path = os.getcwd() + "\\GBM_model_python_18_07_2018"

# Load it back to use it
saved_model = h2o.load_model(path = model_path)
# Write model out
variable_importance = saved_model._model_json['output']['variable_importances'].as_data_frame()
variable_importance.to_csv("variable_importance.csv")
# Write the entire results file: app_test + prediction field
results_path = os.getcwd()[:-11] + "Data\\results.csv"
output_data = app_test.cbind(results[2])
h2o.export_file(output_data, results_path, force=True)

# Load resutls back, extract only the required columns for submission
output_data = h2o.import_file("../Data/results.csv")
submission = output_data[:,["SK_ID_CURR","p1"]]
#submission = submission.rename(columns={'p1': 'TARGET'})
h2o.export_file(submission, os.getcwd() + "\\submission.csv", force=True)


h2o.cluster().shutdown()
# Currently you need to rename the TARGET column from p1, don't know how to
# change column name yet

# This scores an AUC of 0.738

