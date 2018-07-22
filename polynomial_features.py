# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 21:40:05 2018
This file takes the training and test data frames, generate polynomial features,
then save them back as data.

It seems like this approach is not particularly useful
@author: Nathan Chen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Imputer
import os
import gc

# Print current working directory as a check
os.getcwd()

# Load data
app_train = pd.read_csv("../Data/application_train.csv")
app_test =  pd.read_csv("../Data/application_test.csv")
model_path = os.getcwd() + "\\GBM_model_python_18_07_2018"
app_train.columns
print(pd.DataFrame(app_train.columns).to_string())
print(pd.DataFrame(app_test.columns).to_string())

# Make a new dataframe for polynomial features 
poly_features_train = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', "TARGET"]]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
app_test = app_test.drop(columns = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'])
app_train = app_train.drop(columns = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']) 

# imputer for handling missing values, otherwise cannot generate polynomial features
# because of all the NA in the data
imputer = Imputer(strategy = 'median')

# Save the "TARGET" (y variable) because we need to drop it when generating
# polynomial featuers
poly_target = poly_features_train['TARGET']
poly_features_train = poly_features_train.drop(columns = ['TARGET'])

# Need to impute missing values
poly_features_train = imputer.fit_transform(poly_features_train)
poly_features_test = imputer.transform(poly_features_test)

# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 2)

# Train the polynomial features
poly_transformer.fit(poly_features_train)

# Transform the features
poly_features_train = poly_transformer.transform(poly_features_train)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features_train.shape)

# Check the created fields
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])

# Create a dataframe of the features 
poly_features_train = pd.DataFrame(poly_features_train, 
                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3']))
# Add in the target
poly_features_train['TARGET'] = poly_target
poly_features_train.loc[:, 'TARGET']

# Put test features into dataframe
poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']))
gc.collect()
poly_features_train = poly_features_train.loc[:, poly_features_train.columns != "1"]
poly_features_test = poly_features_test.loc[:, poly_features_test.columns != "1"]

# Merge polynomial features into training dataframe  
poly_features_train['SK_ID_CURR'] = app_train['SK_ID_CURR']                                                                            
app_train_poly = app_train.merge(poly_features_train, on = 'SK_ID_CURR', how = 'left')

# Merge polnomial features into testing dataframe
poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')
print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape:  ', app_test_poly.shape)

# Delete some variables and call garbage collection to reduce RAM usage
del poly_features_train, poly_features_test, app_test, app_train

# Align the data frames
app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)
print("Training data is now of shape: ",app_train_poly.shape)
print("Test data is now of shape: ", app_test_poly.shape)


# Put the TARGET back in since it was lost in the alignment process
app_train_poly['TARGET'] = poly_target

# Save as data for use later
app_train_poly = app_train_poly.rename(columns={'EXT_SOURCE_1_y': 'EXT_SOURCE_1', 
                                                'EXT_SOURCE_2_y': 'EXT_SOURCE_2',
                                                'EXT_SOURCE_3_y': 'EXT_SOURCE_3'})
app_test_poly = app_test_poly.rename(columns={'EXT_SOURCE_1_y': 'EXT_SOURCE_1', 
                                                'EXT_SOURCE_2_y': 'EXT_SOURCE_2',
                                                'EXT_SOURCE_3_y': 'EXT_SOURCE_3'})
app_train_poly.to_csv("../Data/app_train_poly.csv", index=False)
app_test_poly.to_csv("../Data/app_test_poly.csv", index=False)
app_train_poly.columns
app_test_poly.columns
# Delete these large data frames and call garbage collection to reduce RAM usage
del app_train_poly, app_test_poly
gc.collect()
