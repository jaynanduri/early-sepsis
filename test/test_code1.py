
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import numpy as np
from collections import Counter
import random
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def load_train_files():

        i=0
        train_files = []
        path = '/Users/pavithra/Desktop/xyz2/dags/data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setA'
        for f in os.listdir(path):
            if i>200000:
                break
            if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
                train_files.append(f)
                i=i+1

        train_df = pd.DataFrame()
        path = '/Users/pavithra/Desktop/xyz2/dags/data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setA'
        for filename in os.listdir(path):
            if filename.endswith('.psv') and filename in train_files:
                file_path = os.path.join(path, filename)
                data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
                file_id = os.path.splitext(filename)[0]
                file_id=int(file_id[1:])
                data['id'] = file_id
                print(file_id)
                train_df = pd.concat([train_df, data], ignore_index=True)

        tr_last_column = train_df.pop(train_df.columns[-1])  # Remove the last column
        train_df.insert(0, tr_last_column.name, tr_last_column)
        #print(train_df.columns)
        print(train_df)
        train_df.to_csv('train.csv')
        #train_df.to_pickle('1_train_df.pkl')
load_train_files()
def load_test_file():
        i=0
        test_files = []
        path = '/Users/pavithra/Desktop/xyz2/dags/data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setB'
        for f in os.listdir(path):
            if i>1000000:
                break
            if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
                test_files.append(f)
                i=i+1

        test_df = pd.DataFrame()
        path = '/Users/pavithra/Desktop/xyz2/dags/data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setB'
        for filename in os.listdir(path):
            if filename.endswith('.psv') and filename in test_files:
                file_path = os.path.join(path, filename)
                data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
                file_id = os.path.splitext(filename)[0]
                file_id=int(file_id[1:])
                data['id'] = file_id
                print(file_id)
                test_df = pd.concat([test_df, data], ignore_index=True)

        ts_last_column = test_df.pop(test_df.columns[-1])  # Remove the last column
        test_df.insert(0, ts_last_column.name, ts_last_column)
        test_df.to_csv('test.csv')
        #test_df.to_pickle('1_test_df.pkl')

#load_test_file()
def feature_engineering():


    with open('/Users/pavithra/Desktop/xyz/1_train_df.pkl', 'rb') as file:
        train_df = pickle.load(file)

    with open('/Users/pavithra/Desktop/xyz/1_test_df.pkl', 'rb') as file:
        test_df = pickle.load(file)

    train_df = pickle.loads(train_data)
    train_df = train_df.drop(train_df.columns[[37,38]], axis=1)
    column_index_to_encode = 36
    encoded_columns = pd.get_dummies(train_df.iloc[:, column_index_to_encode], prefix='column_36')
    train_df.drop(train_df.columns[column_index_to_encode], axis=1, inplace=True)
    train_df = pd.concat([train_df, encoded_columns], axis=1)
    train_df = train_df.drop('EtCO2', axis=1)

    test_df = pickle.loads(test_data)
    test_df = test_df.drop(test_df.columns[[37,38]], axis=1)
    encoded_columns = pd.get_dummies(test_df.iloc[:, column_index_to_encode], prefix='column_36')
    test_df.drop(test_df.columns[column_index_to_encode], axis=1, inplace=True)
    test_df = pd.concat([test_df, encoded_columns], axis=1)
    test_df = test_df.drop('EtCO2', axis=1)

    threshold = 1.0  # 90%
    threshold_count = len(train_df) * threshold
    columns_to_drop = train_df.columns[train_df.isnull().sum() >= threshold_count]
    train_df.drop(columns=columns_to_drop, inplace=True, axis=1)
    test_df.drop(columns=columns_to_drop, inplace=True, axis=1)

    if len(train_df) > 1:
        mask = train_df.groupby('id')['id'].transform('count') >= 15
        train_df = train_df[mask]
        train_df = train_df.groupby('id').apply(lambda x: x.ffill()).reset_index(drop=True)
    

    if len(test_df) > 1:
        test_df = test_df.groupby('id').apply(lambda x: x.ffill()).reset_index(drop=True)

    mean_values = train_df.mean()
    mean_values.to_csv('/Users/pavithra/Desktop/xyz/dags/mean_values.csv')
    test_df.fillna(mean_values, inplace=True)    
    train_df.fillna(mean_values, inplace=True)
    
    train_df.to_pickle('/Users/pavithra/Desktop/xyz/dags/m_train_df.pkl')
    test_df.to_pickle('/Users/pavithra/Desktop/xyz/dags/m_test_df.pkl')

    

    

#feature_engineering()
        
