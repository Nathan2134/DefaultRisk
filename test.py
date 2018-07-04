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

# List files available
print(os.listdir("./"))

# Training data
app_train = pd.read_csv('./application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()

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
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)