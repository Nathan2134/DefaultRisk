# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 21:00:58 2018
Takes some extra data and attach them onto the training and test data frames,
grouping the extra data by ID: SK_ID_CURR 
@author: Nathan Chen
"""

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 
import gc

# Group bureau.csv with bureau_balance.csv
bureau_balance = pd.read_csv('../Data/bureau_balance.csv')
bureau_balance.columns

bureau_balance[["SK_ID_BUREAU"]]
bureau = pd.read_csv('../Data/bureau.csv')
bureau_balance = pd.DataFrame(bureau_balance.groupby(["STATUS"], as_index = False).mean())
bureau = bureau.merge(bureau_balance, on = "SK_ID_BUREAU", how = "left")
# Input data
input_data = pd.read_csv('../Data/bureau.csv')
app_train = pd.read_csv("../Data/application_train.csv")
app_test = pd.read_csv("../Data/application_test.csv")
print('Training data shape: ', input_data.shape)
# (1716428, 17) for bureau data

print("Dimensions of the data to be added: ", input_data.groupby(["SK_ID_CURR"])
.mean().shape)
# Group by mean
input_data = bureau.groupby(["SK_ID_CURR"]).mean()

# Attach back to the training and test data
app_train = app_train.merge(input_data, on = 'SK_ID_CURR', how = 'left') 
app_test = app_test.merge(input_data, on = 'SK_ID_CURR', how = 'left') 

# Save
app_train.to_csv("../Data/app_train_with_extra.csv", index = False)
app_test.to_csv("../Data/app_test_with_extra.csv", index = False)
gc.collect()

gc.collect()