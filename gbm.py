# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 11:40:59 2018

This file applies the Gradient Boosting Machine algorithm from H2O.ai
on the training data.

@author: Nathan Chen
"""

import numpy as np
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import h2o
import os

# Check current working directory
os.getcwd()

h2o.init()
# Load training data
app_train = h2o.import_file(path = '../Data/application_train.csv')
""" Ensure to make the y variable, "TARGET", a factor, since we are doing classification
otherwise the model interprets y as between 0 and 1, and
prediction results may return a negative number  """
app_train[["TARGET"]] = app_train[["TARGET"]].asfactor()
# Specify the model
gbm_model = H2OGradientBoostingEstimator()
# Train the model, put all parameters other than "TARGET" in, the first column is
# "TARGET", the y variable
gbm_model.train(x=app_train.names[2:], y = app_train.names[1], training_frame = app_train)
# Load test set
app_test = h2o.import_file(path = '../Data/application_test.csv')
results = gbm_model.predict(test_data = app_test)
# View the model if you want
gbm_model.show()
print(results.describe())

# Save the model
model_path = h2o.save_model(gbm_model, path= os.getcwd())

# Use this if you gave your model a custom name
# model_path = os.getcwd() + "\\GBM_model_python_16_07_2018"

# Load it back to use it
saved_model = h2o.load_model(path = model_path)

# Write the entire results file: app_test + prediction field
results_path = os.getcwd()[:-11] + "Data\\results.csv"
output_data = app_test.cbind(results[2])
h2o.export_file(output_data, results_path, force=True)

# Load resutls back, extract only the required columns for submission
output_data = h2o.import_file("../Data/results.csv")
submission = output_data[:,["SK_ID_CURR","p1"]]
h2o.export_file(submission, os.getcwd() + "\\submission.csv", force=True)

# Currently you need to rename the TARGET column from p1, don't know how to
# change column name yet

# This scores an AUC of 0.738


