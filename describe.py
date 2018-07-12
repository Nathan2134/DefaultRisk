#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 22:40:52 2018

@author: NanChen
"""

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
# File system manangement
import os
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# List date files available
# Please note that this assumes you place the csv data files in a folder 
# called "Data", just outside of this "DefaultRisk" folder. So the root folder
# that contains "DefaultRisk" would also contain "Data".
print(os.listdir("../Data"))

# Training data
app_train = pd.read_csv('../Data/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()

# Histogram
#x = app_train['TARGET'].describe()
#app_train['TARGET'].nunique()
text_stat = []
num_stat = []
for col in app_train:
    if app_train[col].dtype == 'object': 
        unique_val = app_train[col].unique().tolist()
        val_counts = app_train[col].value_counts().tolist()
        text_stat.append([col, unique_val,val_counts, app_train[col].nunique()])
    # If numerical
    else:
        col_stat = app_train[col].describe()
        col_mean = col_stat["mean"]
        col_std = col_stat["std"]
        col_min = col_stat["min"]
        col_25 = col_stat["25%"]
        col_50 = col_stat["50%"]
        col_75 = col_stat["75%"]
        col_max = col_stat["max"]
        n_unique = app_train[col].nunique()
        n_unique_w_na = len(list(app_train[col].unique()))
        # This is 0 then na does no exist
        na_exist = n_unique_w_na - n_unique
        miss_val = app_train[col].isnull().sum()
        miss_val_perc = app_train[col].isnull().sum() / len(app_train[col])
        data_type = app_train[col].dtypes
        num_stat.append([col, col_mean, col_std, col_min, col_25, col_50, col_75, 
                         col_max, n_unique, n_unique_w_na, na_exist, miss_val,
                         miss_val_perc, data_type])
text_stat = pd.DataFrame(text_stat, columns=["column_name","unique valueus", "value counts", 
                                            "number of unique values"])
text_stat.to_csv("text_stat.csv", index = False)
num_stat = pd.DataFrame(num_stat, columns=["column_name", "mean", "std", "min", "25%", "50%",
                                           "75%","max", "number of unique val",
                                           "number of unique with na", "na_exist",
                                           "number of na", "percentage of na",
                                           "data_type"])
num_stat.to_csv("num_stat.csv", index = False)
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist();