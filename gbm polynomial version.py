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
app_train = pd.read_csv("../Data/app_train_poly.csv")
app_train = h2o.H2OFrame(app_train)
print(pd.DataFrame(app_train.columns).to_string())
# Ensure to make the y variable a factor, otherwise the training result may
# return a negative number
app_train[["TARGET"]] = app_train[["TARGET"]].asfactor()
app_train[['NONLIVINGAREA_MEDI']] = app_train[['NONLIVINGAREA_MEDI']].asnumeric()
app_train[['NONLIVINGAPARTMENTS_MEDI']] = app_train[['NONLIVINGAPARTMENTS_MEDI']].asnumeric()
app_train[['LIVINGAPARTMENTS_MEDI']] = app_train[['LIVINGAPARTMENTS_MEDI']].asnumeric()
app_train[['LANDAREA_MEDI']] = app_train[['LANDAREA_MEDI']].asnumeric()
app_train[['COMMONAREA_MEDI']] = app_train[['COMMONAREA_MEDI']].asnumeric()
app_train[['BASEMENTAREA_MEDI']] = app_train[['BASEMENTAREA_MEDI']].asnumeric()
app_train[['NONLIVINGAREA_MODE']] = app_train[['NONLIVINGAREA_MODE']].asnumeric()
app_train[['NONLIVINGAPARTMENTS_MODE']] = app_train[['NONLIVINGAPARTMENTS_MODE']].asnumeric()
app_train[['LIVINGAPARTMENTS_MODE']] = app_train[['LIVINGAPARTMENTS_MODE']].asnumeric()
app_train[['LANDAREA_MODE']] = app_train[['LANDAREA_MODE']].asnumeric()
app_train[['COMMONAREA_MODE']] = app_train[['COMMONAREA_MODE']].asnumeric()
app_train[['BASEMENTAREA_MODE']] = app_train[['BASEMENTAREA_MODE']].asnumeric()
app_train[['NONLIVINGAREA_AVG']] = app_train[['NONLIVINGAREA_AVG']].asnumeric()
app_train[['NONLIVINGAPARTMENTS_AVG']] = app_train[['NONLIVINGAPARTMENTS_AVG']].asnumeric()
app_train[['LIVINGAPARTMENTS_AVG']] = app_train[['LIVINGAPARTMENTS_AVG']].asnumeric()
app_train[['LANDAREA_AVG']] = app_train[['LANDAREA_AVG']].asnumeric()
app_train[['FLOORSMIN_AVG']] = app_train[['FLOORSMIN_AVG']].asnumeric()
app_train[['ELEVATORS_AVG']] = app_train[['ELEVATORS_AVG']].asnumeric()
app_train[['COMMONAREA_AVG']] = app_train[['COMMONAREA_AVG']].asnumeric()
app_train[['BASEMENTAREA_AVG']] = app_train[['BASEMENTAREA_AVG']].asnumeric()
app_train[['OWN_CAR_AGE']] = app_train[['OWN_CAR_AGE']].asnumeric()

# Specify the model
gbm_model = H2OGradientBoostingEstimator()
# Train the model, put all parameters other than "TARGET" in, the first column is
# "TARGET", the y variable
gbm_model.train(x=app_train.names[:-1], y = app_train.names[-1], training_frame = app_train)
# Load test set
#app_test = h2o.import_file(path = '../Data/app_test_poly.csv')
app_test = pd.read_csv("../Data/app_test_poly.csv")
app_test = h2o.H2OFrame(app_test)

app_test[['NONLIVINGAREA_MEDI']] = app_test[['NONLIVINGAREA_MEDI']].asnumeric()
app_test[['NONLIVINGAPARTMENTS_MEDI']] = app_test[['NONLIVINGAPARTMENTS_MEDI']].asnumeric()
app_test[['LIVINGAPARTMENTS_MEDI']] = app_test[['LIVINGAPARTMENTS_MEDI']].asnumeric()
app_test[['LANDAREA_MEDI']] = app_test[['LANDAREA_MEDI']].asnumeric()
app_test[['COMMONAREA_MEDI']] = app_test[['COMMONAREA_MEDI']].asnumeric()
app_test[['BASEMENTAREA_MEDI']] = app_test[['BASEMENTAREA_MEDI']].asnumeric()
app_test[['NONLIVINGAREA_MODE']] = app_test[['NONLIVINGAREA_MODE']].asnumeric()
app_test[['NONLIVINGAPARTMENTS_MODE']] = app_test[['NONLIVINGAPARTMENTS_MODE']].asnumeric()
app_test[['LIVINGAPARTMENTS_MODE']] = app_test[['LIVINGAPARTMENTS_MODE']].asnumeric()
app_test[['LANDAREA_MODE']] = app_test[['LANDAREA_MODE']].asnumeric()
app_test[['COMMONAREA_MODE']] = app_test[['COMMONAREA_MODE']].asnumeric()
app_test[['BASEMENTAREA_MODE']] = app_test[['BASEMENTAREA_MODE']].asnumeric()
app_test[['NONLIVINGAREA_AVG']] = app_test[['NONLIVINGAREA_AVG']].asnumeric()
app_test[['NONLIVINGAPARTMENTS_AVG']] = app_test[['NONLIVINGAPARTMENTS_AVG']].asnumeric()
app_test[['LIVINGAPARTMENTS_AVG']] = app_test[['LIVINGAPARTMENTS_AVG']].asnumeric()
app_test[['LANDAREA_AVG']] = app_test[['LANDAREA_AVG']].asnumeric()
app_test[['FLOORSMIN_AVG']] = app_test[['FLOORSMIN_AVG']].asnumeric()
app_test[['ELEVATORS_AVG']] = app_test[['ELEVATORS_AVG']].asnumeric()
app_test[['COMMONAREA_AVG']] = app_test[['COMMONAREA_AVG']].asnumeric()
app_test[['BASEMENTAREA_AVG']] = app_test[['BASEMENTAREA_AVG']].asnumeric()
app_test[['OWN_CAR_AGE']] = app_test[['OWN_CAR_AGE']].asnumeric()

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
h2o.export_file(submission, os.getcwd() + "\\submission.csv", force=True)


h2o.cluster().shutdown()
# Currently you need to rename the TARGET column from p1, don't know how to
# change column name yet

# This scores an AUC of 0.738

