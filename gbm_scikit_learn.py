# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 20:51:05 2018

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 11:40:59 2018

This file applies the Gradient Boosting Machine algorithm from H2O.ai
on the training data.

@author: Nathan Chen
"""

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier

app_train = pd.read_csv("../Data/app_train_with_extra.csv")
app_train.dtypes
gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=7).fit(app_train.iloc[:, 2:],  app_train.iloc[:,1])
feature_importance_values = gbm_model.feature_importances_
feature_importances_domain = pd.DataFrame({'feature': domain_features_names, 'importance': feature_importance_values_domain})