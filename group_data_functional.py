# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 21:25:05 2018

@author: Admin
"""

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 
import gc

def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical

# Group bureau.csv with bureau_balance.csv
bureau_balance = pd.read_csv('../Data/bureau_balance.csv')
bureau = pd.read_csv('../Data/bureau.csv')
bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_counts = pd.DataFrame(bureau_counts)
bureau_counts.to_csv("bureau_counts.csv")
bureau_balance = pd.DataFrame(bureau_balance.groupby(["SK_ID_BUREAU"], as_index = False).mean())
bureau = bureau.merge(bureau_balance, on = "SK_ID_BUREAU", how = "left")
