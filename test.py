"""
Created on Tue Jul  3 22:19:58 2018
@author: Nathan

This is an initial test for data processing, exploration and cleaning
It is based on the introductory code of Will Koehrsen.
Note that the this file should be set to run from the directory it is in
for the system path to work properly.
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

# Testing data features
app_test = pd.read_csv('../Data/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()

# Histogram
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist();

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results, axis = 1 means columns
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

# Missing values statistics
missing_values = missing_values_table(app_train)
missing_values.head(20)

# Number of each type of column
app_train.dtypes.value_counts()

# Number of unique classes in each object column, axis = 0 means row
# 'object' means text string field here
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


le = LabelEncoder()
le_count = 0

# Iterate through the columns and encode columns with 2 or fewer categories
for col in app_train:
    if app_train[col].dtype == 'object':  
        print(col)
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
# NAME_CONTRACT_TYPE, FLAG_OWN_CAR, FLAG_OWN_REALTY encoded.

# Note that nunique and len(list(pandas.series.unique)) actually give two 
# different results. Check this later.
#len(list(app_train.EMERGENCYSTATE_MODE.unique()))
#app_train.EMERGENCYSTATE_MODE.nunique()

# One-hot encoding for columns with 3 or more categories
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in 
# both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

(app_train['DAYS_BIRTH'] / -365).describe()
app_train['DAYS_EMPLOYED'].describe()
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');

anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment out of %d total days' % 
      (len(anom),len(app_train['DAYS_EMPLOYED'])))

# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');