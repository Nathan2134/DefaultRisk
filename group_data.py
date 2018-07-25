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
bureau = pd.read_csv('../Data/bureau.csv')
bureau_balance = pd.DataFrame(bureau_balance.groupby(["SK_ID_BUREAU"], as_index = False).mean())
bureau = bureau.merge(bureau_balance, on = "SK_ID_BUREAU", how = "left")
print("Dimensions of the data to be added: ", bureau.shape)
# (1716428, 19) for combined bureau and bureau balance data

# Training and test data
app_train = pd.read_csv("../Data/application_train.csv")
app_test = pd.read_csv("../Data/application_test.csv")

# Group by mean
bureau = bureau.groupby(["SK_ID_CURR"], as_index = False).mean()

# Attach back to the training and test data
app_train = app_train.merge(bureau, on = 'SK_ID_CURR', how = 'left') 
app_test = app_test.merge(bureau, on = 'SK_ID_CURR', how = 'left') 

# Save
app_train.to_csv("../Data/app_train_with_extra.csv", index = False)
app_test.to_csv("../Data/app_test_with_extra.csv", index = False)
gc.collect()

# Group previous_application.csv with POS_CASH_balance.csv
previous_application = pd.read_csv('../Data/previous_application.csv')
pos_cash = pd.read_csv('../Data/POS_CASH_balance.csv')
pos_cash = pos_cash.drop(columns=['SK_ID_CURR'])
pos_cash = pd.DataFrame(pos_cash.groupby(["SK_ID_PREV"], as_index = False).mean())
previous_application = previous_application.merge(pos_cash, on = "SK_ID_PREV", how = "left")
print("Dimensions of the data to be added: ", previous_application.shape)
# (1716428, 19) for previous_application combined with pos cash data

# Read training and test data
app_train = pd.read_csv("../Data/app_train_with_extra.csv")
app_test = pd.read_csv("../Data/app_test_with_extra.csv")

# Group by mean
previous_application = previous_application.groupby(["SK_ID_CURR"], as_index = False).mean()

# Attach back to the training and test data
app_train = app_train.merge(previous_application, on = "SK_ID_CURR", how = 'left') 
app_test = app_test.merge(previous_application, on = "SK_ID_CURR", how = 'left') 

# Group previous_application.csv with installments_payments.csv
installments = pd.read_csv('../Data/installments_payments.csv')
installments = installments.drop(columns=['SK_ID_CURR'])
installments = pd.DataFrame(installments.groupby(["SK_ID_PREV"], as_index = False).mean())
previous_application = previous_application.merge(installments, on = "SK_ID_PREV", how = "left")
print("Dimensions of the data to be added: ", previous_application.shape)

# Group by mean
previous_application = previous_application.groupby(["SK_ID_CURR"], as_index = False).mean()

# Attach back to the training and test data
app_train = app_train.merge(previous_application, on = "SK_ID_CURR", how = 'left') 
app_test = app_test.merge(previous_application, on = "SK_ID_CURR", how = 'left') 


# Group previous_application.csv with credit_card_balance.csv
credit_card = pd.read_csv('../Data/credit_card_balance.csv')
credit_card = credit_card.drop(columns=['SK_ID_CURR'])
credit_card = pd.DataFrame(credit_card.groupby(["SK_ID_PREV"], as_index = False).mean())
previous_application = previous_application.merge(credit_card, on = "SK_ID_PREV", how = "left")
print("Dimensions of the data to be added: ", previous_application.shape)

# Group by mean
previous_application = previous_application.groupby(["SK_ID_CURR"], as_index = False).mean()

# Attach back to the training and test data
app_train = app_train.merge(previous_application, on = "SK_ID_CURR", how = 'left') 
app_test = app_test.merge(previous_application, on = "SK_ID_CURR", how = 'left') 

# Save
app_train.to_csv("../Data/app_train_with_extra.csv", index = False)
app_test.to_csv("../Data/app_test_with_extra.csv", index = False)
gc.collect()
