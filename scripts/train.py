# import libraries
## general
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
## custom
from util_funcs import process_data, split_data
from neural_networks import SimpleNN, ComplexNN
## models
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


#----------------------------------------------------------------------

# Set up logging to a file
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
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

weights_filename = 'logs/feature_weights.txt'

#----------------------------------------------------------------------

### GENDER DATA ###

# initiate models
## retrieve model parameters from log
lasso_reg_G = Lasso()
ridge_reg_G = Ridge()
svr_reg_G = SVR()
nn1_G = SimpleNN(input_size=X_train_G.shape[1], hidden_size=10, output_size=1)
nn2_G = ComplexNN(input_size=X_train_G.shape[1], 
                  hidden_size=[X_train_G.shape[1], 45, 22, 11], 
                  output_size=1)

# train models
print('Beginning model training...')

## lists for storing performance metrics
mseHistory_lasso_G = []
mseHistory_ridge_G = []
mseHistory_svr_G = []
maeHistory_lasso_G = []
maeHistory_ridge_G = []
maeHistory_svr_G = []

for epoch in range(100):
    lasso_reg_G.fit(X_train_G, y_train_G)
    ridge_reg_G.fit(X_train_G, y_train_G)
    svr_reg_G.fit(X_train_G, y_train_G)

    # make predictions
    y_pred_lasso_G = lasso_reg_G.predict(X_test_G)
    y_pred_ridge_G = ridge_reg_G.predict(X_test_G)
    y_pred_svr_G = svr_reg_G.predict(X_test_G)

    # calculate performance metrics
    ## MSE
    mse_lasso_G = mean_squared_error(y_test_G, y_pred_lasso_G)
    mseHistory_lasso_G.append(mse_lasso_G)
    mse_ridge_G = mean_squared_error(y_test_G, y_pred_ridge_G)
    mseHistory_ridge_G.append(mse_ridge_G)
    mse_svr_G = mean_squared_error(y_test_G, y_pred_svr_G)
    mseHistory_svr_G.append(mse_svr_G)
    # MAE
    mae_lasso_G = mean_absolute_error(y_test_G, y_pred_lasso_G)
    maeHistory_lasso_G.append(mae_lasso_G)
    mae_ridge_G = mean_absolute_error(y_test_G, y_pred_ridge_G)
    maeHistory_ridge_G.append(mae_ridge_G)
    mae_svr_G = mean_absolute_error(y_test_G, y_pred_svr_G)
    maeHistory_svr_G.append(mae_svr_G)
    
lasso_weights_G = lasso_reg_G.coef_
ridge_weights_G = ridge_reg_G.coef_
svr_weights_G = svr_reg_G.coef_

with open(weights_filename, 'w') as f:
    f.write("Weights for Lasso Regression on Gender Inclusive Data")
    f.write("------------------------------------------------------")
    for i, weight in enumerate(lasso_weights_G):
        f.write("Feature {}: {}\n".format(i+1, weight))
    f.write("Weights for Ridge Regression on Gender Inclusive Data")
    f.write("------------------------------------------------------")
    for i, weight in enumerate(ridge_weights_G):
        f.write("Feature {}: {}\n".format(i+1, weight))
    f.write("Weights for SVR Regression on Gender Inclusive Data")
    f.write("------------------------------------------------------")
    for i, weight in enumerate(svr_weights_G):
        f.write("Feature {}: {}\n".format(i+1, weight))

mseHistory_nn1_G, maeHistory_nn1_G = nn1_G.train(X_train_G, y_train_G, learning_rate=0.01, epochs=100)
mseHistory_nn2_G, maeHistory_nn2_G = nn2_G.train(X_train_G, y_train_G, learning_rate=0.01, epochs=100)

nn1_input_weights_G, nn1_output_weights_G = nn1_G.get_weights()
nn2_weights_G = nn2_G.get_weights()

with open(weights_filename, 'w') as f:
    f.write("Weights for Simple NN on Gender Inclusive Data")
    f.write("------------------------------------------------------")
    f.write("Weights input-hidden:", nn1_input_weights_G)
    f.write("Weights hidden-output:", nn1_output_weights_G)
    f.write("Weights for Complex NN on Gender Inclusive Data")
    f.write("------------------------------------------------------")
    for i, weight in enumerate(nn2_weights_G):
        f.write(f"Weights of layer {i+1}:", weight)
    
learning_history_G = {'epoch':range(1,101),
                      'mse_lasso':mseHistory_lasso_G,
                      'mse_ridge':mseHistory_ridge_G,
                      'mse_svr':mseHistory_svr_G,
                      'mse_nn1':mseHistory_nn1_G,
                      'mse_nn2':mseHistory_nn2_G,
                      'mae_lasso':maeHistory_lasso_G,
                      'mae_ridge':maeHistory_ridge_G,
                      'mae_svr':maeHistory_svr_G,
                      'mae_nn1':maeHistory_nn1_G,
                      'mae_nn2':maeHistory_nn2_G
                     }

df_learning_history_G = pd.DataFrame(learning_history_G)
csv_filename = 'logs/training_history_gender.csv'
df_learning_history_G.to_csv(csv_filename, index=False)

#-------------------------------------------------------------------

### NO GENDER DATA ###

# initiate models
## retrieve model parameters from log
lasso_reg_NG = Lasso()
ridge_reg_NG = Ridge()
svr_reg_NG = SVR()
nn1_NG = SimpleNN(input_size=X_train_NG.shape[1], hidden_size=10, output_size=1)
nn2_NG = ComplexNN(input_size=X_train_NG.shape[1], 
                  hidden_size=[X_train_NG.shape[1], 45, 22, 11], 
                  output_size=1)

# train models
print('Beginning model training...')

## lists for storing performance metrics
mseHistory_lasso_NG = []
mseHistory_ridge_NG = []
mseHistory_svr_NG = []
maeHistory_lasso_NG = []
maeHistory_ridge_NG = []
maeHistory_svr_NG = []

for epoch in range(100):
    lasso_reg_NG.fit(X_train_NG, y_train_NG)
    ridge_reg_NG.fit(X_train_NG, y_train_NG)
    svr_reg_NG.fit(X_train_NG, y_train_NG)

    # make predictions
    y_pred_lasso_NG = lasso_reg_NG.predict(X_test_NG)
    y_pred_ridge_NG = ridge_reg_NG.predict(X_test_NG)
    y_pred_svr_NG = svr_reg_NG.predict(X_test_NG)

    # calculate performance metrics
    ## MSE
    mse_lasso_NG = mean_squared_error(y_test_NG, y_pred_lasso_NG)
    mseHistory_lasso_NG.append(mse_lasso_NG)
    mse_ridge_NG = mean_squared_error(y_test_NG, y_pred_ridge_NG)
    mseHistory_ridge_NG.append(mse_ridge_NG)
    mse_svr_NG = mean_squared_error(y_test_NG, y_pred_svr_NG)
    mseHistory_svr_NG.append(mse_svr_NG)
    # MAE
    mae_lasso_NG = mean_absolute_error(y_test_NG, y_pred_lasso_NG)
    maeHistory_lasso_NG.append(mae_lasso_NG)
    mae_ridge_NG = mean_absolute_error(y_test_NG, y_pred_ridge_NG)
    maeHistory_ridge_NG.append(mae_ridge_NG)
    mae_svr_NG = mean_absolute_error(y_test_NG, y_pred_svr_NG)
    maeHistory_svr_NG.append(mae_svr_NG)

lasso_weights_NG = lasso_reg_NG.coef_
ridge_weights_NG = ridge_reg_NG.coef_
svr_weights_NG = svr_reg_NG.coef_

with open(weights_filename, 'w') as f:
    f.write("Weights for Lasso Regression on Gender Exclusive Data")
    f.write("------------------------------------------------------")
    for i, weight in enumerate(lasso_weights_NG):
        f.write("Feature {}: {}\n".format(i+1, weight))
    f.write("Weights for Ridge Regression on Gender Exclusive Data")
    f.write("------------------------------------------------------")
    for i, weight in enumerate(ridge_weights_NG):
        f.write("Feature {}: {}\n".format(i+1, weight))
    f.write("Weights for SVR Regression on Gender Exclusive Data")
    f.write("------------------------------------------------------")
    for i, weight in enumerate(svr_weights_NG):
        f.write("Feature {}: {}\n".format(i+1, weight))
    
mseHistory_nn1_NG, maeHistory_nn1_NG = nn1_NG.train(X_train_NG, y_train_NG, learning_rate=0.01, epochs=100)
mseHistory_nn2_NG, maeHistory_nn2_NG = nn2_NG.train(X_train_NG, y_train_NG, learning_rate=0.01, epochs=100)

nn1_input_weights_NG, nn1_output_weights_NG = nn1_NG.get_weights()
nn2_weights_NG = nn2_NG.get_weights()

with open(weights_filename, 'w') as f:
    f.write("Weights for Simple NN on Gender Exclusive Data")
    f.write("------------------------------------------------------")
    f.write("Weights input-hidden:", nn1_input_weights_NG)
    f.write("Weights hidden-output:", nn1_output_weights_NG)
    f.write("Weights for Complex NN on Gender Exclusive Data")
    f.write("------------------------------------------------------")
    for i, weight in enumerate(nn2_weights_NG):
        f.write(f"Weights of layer {i+1}:", weight)
    
learning_history_NG = {'epoch':range(1,101),
                      'mse_lasso':mseHistory_lasso_NG,
                      'mse_ridge':mseHistory_ridge_NG,
                      'mse_svr':mseHistory_svr_NG,
                      'mse_nn1':mseHistory_nn1_NG,
                      'mse_nn2':mseHistory_nn2_NG,
                      'mae_lasso':maeHistory_lasso_NG,
                      'mae_ridge':maeHistory_ridge_NG,
                      'mae_svr':maeHistory_svr_NG,
                      'mae_nn1':maeHistory_nn1_NG,
                      'mae_nn2':maeHistory_nn2_NG
                     }

df_learning_history_NG = pd.DataFrame(learning_history_NG)
csv_filename = 'logs/training_history_no-gender.csv'
df_learning_history_NG.to_csv(csv_filename, index=False)