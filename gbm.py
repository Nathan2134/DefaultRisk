# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 11:40:59 2018

@author: Admin
"""

import numpy as np
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import h2o

h2o.init()
app_train = h2o.import_file(path = '../Data/application_train.csv')
gbm_model = H2OGradientBoostingEstimator()
gbm_model.train(x=app_train.names[2:], y = app_train.names[1], training_frame = app_train)
app_test = h2o.import_file(path = '../Data/application_test.csv')
results = gbm_model.predict(test_data = app_test)
gbm_model.show()

# model_path = h2o.save_model(gbm_model, path="./")
# You should change this to whatever your directory is
saved_model = h2o.load_model(path = "C:\\Users\\Admin\\Desktop\\Kaggle\\DefaultRisk\\GBM_model_python_16_07_2018")

pd.concat([app_test, results])
h2o.export_file(results, "./results.csv", force=True)
gbm_model.relative_importance()
results.to_csv("gbm-results-v1.csv", index = False)
print(gbm_model.model_performance)
gbm_model.get_params()
gbm