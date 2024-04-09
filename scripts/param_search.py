# import libraries
## general
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import logging
## custom
from util_funcs import process_data, split_data
from neural_networks import SimpleNN
## models
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

#----------------------------------------------------------------------

# Set up logging to a file
logging.basicConfig(filename='logs/param_search.log', level=logging.INFO, format='%(asctime)s - %(message)s')
print('Training is beginning...')

# load data
df = pd.read_csv('/Volumes/Hodges1/2024/Spring/ESE527/Project/data/model-data/cps_ALL.csv')
print('Original data has been loaded...')

# process data
df_temp1 = df.copy()
df_gender = process_data(df_temp1,'gender')
df_temp2 = df.copy()
df_noGender = process_data(df_temp2,'no-gender')

# split data (G = gender, NG = no gender)
X_train_G, X_test_G, y_train_G, y_test_G = split_data(df_gender)
X_train_NG, X_test_NG, y_train_NG, y_test_NG = split_data(df_noGender)
print('Data has been pre-processed...')

#----------------------------------------------------------------------

### GENDER DATA ###

print('Beginning hyperparameter search for gender inclusive data...')

# initiate models
lasso_reg_G = Lasso()
ridge_reg_G = Ridge()
svr_reg_G = SVR()

# parameters to search
lasso_params = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'fit_intercept': [True, False],
    'max_iter': [5000]
}
ridge_params = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],
    'fit_intercept': [True, False],
    'max_iter': [5000]
}
svr_params = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'epsilon': [0.01, 0.1, 1, 10]
}

# search model parameters
lasso_grid_G = GridSearchCV(lasso_reg_G, lasso_params, cv=5)
lasso_grid_G.fit(X_train_G, y_train_G)
ridge_grid_G = GridSearchCV(ridge_reg_G, ridge_params, cv=5)
ridge_grid_G.fit(X_train_G, y_train_G)
svr_grid_G = GridSearchCV(svr_reg_G, svr_params, cv=5)
svr_grid_G.fit(X_train_G, y_train_G)
# log information
logging.info('Best parameters for models using gender inclusive data')
logging.info(f'Lasso regression: {lasso_grid_G.best_params_}')
logging.info(f'Ridge regression: {ridge_grid_G.best_params_}')
logging.info(f'SVR regression: {svr_grid_G.best_params_}')
print('The hyperparameter searches are complete.')

#-------------------------------------------------------------------

### NO GENDER DATA ###

print('Beginning hyperparameter search for gender exclusive data...')

# initiate models
lasso_reg_NG = Lasso()
ridge_reg_NG = Ridge()
svr_reg_NG = SVR()

# parameters to search
lasso_params = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'fit_intercept': [True, False],
    'max_iter': [5000]
}
ridge_params = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],
    'fit_intercept': [True, False],
    'max_iter': [5000]
}
svr_params = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'epsilon': [0.01, 0.1, 1, 10]
}

# search model parameters
lasso_grid_NG = GridSearchCV(lasso_reg_NG, lasso_params, cv=5)
lasso_grid_NG.fit(X_train_NG, y_train_NG)
ridge_grid_NG = GridSearchCV(ridge_reg_NG, ridge_params, cv=5)
ridge_grid_NG.fit(X_train_NG, y_train_NG)
svr_grid_NG = GridSearchCV(svr_reg_NG, svr_params, cv=5)
svr_grid_NG.fit(X_train_NG, y_train_NG)
# log information
logging.info('Best parameters for models using gender exclusive data')
logging.info(f'Lasso regression: {lasso_grid_NG.best_params_}')
logging.info(f'Ridge regression: {ridge_grid_NG.best_params_}')
logging.info(f'SVR regression: {svr_grid_NG.best_params_}')
print('The hyperparameter searches are complete.')

orint('All done! Check the log file for more information.')