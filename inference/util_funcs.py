import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def process_data(df, version='gender'):
    
    # drop columns based on gender inclusion/exclusion
    if version == 'gender':
        df.drop(columns=['YEAR', 'SERIAL', 'MONTH', 'CPSID', 'ASECFLAG',
                          'HFLAG', 'ASECWTH', 'PERNUM', 'CPSIDV', 'CPSIDP',
                          'ASECWT', 'NCHILD', 'NCHLT5', 'WKSWORK2', 'WKSUNEM2',
                          'WKSUNEM1', 'ADJGINC', 'TAXINC'], inplace=True)
    elif version == 'no-gender':
        df.drop(columns=['YEAR', 'SERIAL', 'MONTH', 'CPSID', 'ASECFLAG',
                          'HFLAG', 'ASECWTH', 'PERNUM', 'CPSIDV', 'CPSIDP',
                          'ASECWT', 'SEX', 'NCHILD', 'NCHLT5', 'WKSWORK2',
                         'WKSUNEM2','WKSUNEM1', 'ADJGINC', 
                         'TAXINC'], inplace=True)
    
    # drop rows with NaN/NIU values 
    cols_with_zeros = ['DIFFANY', 'SCHLCOLL', 'CLASSWKR', 'IND', 'OCC',
                       'LABFORCE', 'VETSTAT']
    df = df[~df[cols_with_zeros].isin([0]).any(axis=1)]
    df = df[~df['UHRSWORKT'].isin([999, 997])]
    
    # encode categorical features
    if version == 'gender':
        categorical_columns = ['RELATE', 'SEX', 'RACE', 'MARST', 'VETSTAT',
                               'FAMREL','BPL', 'CITIZEN', 'EMPSTAT', 'LABFORCE', 
                               'OCC', 'IND','CLASSWKR', 'EDUC', 'SCHLCOLL', 
                               'DIFFANY']
    elif version == 'no-gender':
        categorical_columns = ['RELATE', 'RACE', 'MARST', 'VETSTAT',
                               'FAMREL', 'BPL', 'CITIZEN', 'EMPSTAT', 'LABFORCE',
                               'OCC', 'IND','CLASSWKR', 'EDUC', 'SCHLCOLL', 
                               'DIFFANY']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, prefix=None)
    
    # drop features that do not meet requirements
    columns_to_keep1 = ['OCC_2310', 'OCC_3255', 'OCC_4700', 'OCC_4720',
                       'OCC_9130', 'OCC_4760', 'OCC_430']
    columns_to_drop1 = [col for col in df_encoded.columns if col.startswith('OCC_') and col not in columns_to_keep1]
    df_encoded.drop(columns=columns_to_drop1, inplace=True)
    columns_to_keep2 = ['IND_770', 'IND_7860', 'IND_8680', 'IND_8190', 
                       'IND_7870', 'IND_7380', 'IND_9470','IND_8191']
    columns_to_drop2 = [col for col in df_encoded.columns if col.startswith('IND_') and col not in columns_to_keep2]
    df_encoded.drop(columns=columns_to_drop2, inplace=True)
    columns_to_keep3 = ['BPL_9900', 'BPL_20000', 'BPL_52100', 'BPL_51500']
    columns_to_drop3 = [col for col in df_encoded.columns if col.startswith('BPL_') and col not in columns_to_keep3]
    df_encoded.drop(columns=columns_to_drop3, inplace=True)
    columns_to_keep4 = ['RACE_100', 'RACE_200', 'RACE_651']
    columns_to_drop4 = [col for col in df_encoded.columns if col.startswith('RACE_') and col not in columns_to_keep4]
    df_encoded.drop(columns=columns_to_drop4, inplace=True)
    
    # drop NaN data
    df_encoded = df_encoded[df_encoded['UHRSWORKLY'] != 999]
    
    # outlier detection
    cols_to_check = ['AGE', 'UHRSWORKT', 'WKSWORK1', 'UHRSWORKLY', 'INCTOT']
    z_scores = stats.zscore(df_encoded[cols_to_check])
    threshold = 3
    outliers_mask = (z_scores > threshold).any(axis=1)
    df_cleaned = df_encoded[~outliers_mask]
    
    # drop perfectly correlated features
    if version == 'gender':
        df_final = df_cleaned.drop(["SEX_2", "VETSTAT_2", "EMPSTAT_12", "DIFFANY_2", "LABFORCE_2"], axis=1)
    elif version == 'no-gender':
        df_final = df_cleaned.drop(["VETSTAT_2", "EMPSTAT_12", "DIFFANY_2", "LABFORCE_2"], axis=1)
    
    # normalize ordinal features
    # UPDATE no longer normalizing INCTOT because that is our target variable
    df_final_nor = df_final.copy()
    cols_to_normalize = ['AGE', 'UHRSWORKT', 'WKSWORK1', 'UHRSWORKLY']
    scaler = MinMaxScaler()
    df_final_nor[cols_to_normalize] = scaler.fit_transform(df_final[cols_to_normalize])
    
    return df_final_nor


def split_data(df):
    
    feature_columns = df.loc[:, df.columns != 'INCTOT'].columns
    X = df[feature_columns]
    y = df['INCTOT']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    train_x = X_train.values
    train_y = y_train.values
    test_x = X_test.values
    test_y = y_test.values
    
    return train_x, test_x, train_y, test_y