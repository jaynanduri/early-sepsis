import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np, os, sys
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
import sys
import warnings
from urllib.parse import urlparse
import pickle
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def delete_last_5_records(group):
    return group.iloc[:-5]


def select_last_50_or_all(group):
    return group.tail(50) if len(group) >= 50 else group


def pad_trunc(data, max_len):
    new_data = []
    zero_vec = [0.0] * len(data[0][0])  # Initialize zero vector
    
    for sample in data:
        if len(sample) > max_len:  # truncate
            temp = sample[:max_len]
        elif len(sample) < max_len:
            temp = sample
            number_additional_elements = max_len - len(sample)
            for _ in range(number_additional_elements):
                temp.append(zero_vec)  # append zero vectors at the end
        else:
            temp = sample
    
        new_data.append(temp)
    
    return new_data
    
    

def create_sequence_tr_test_data(df, feature_columns, y_label='y'):
    id_list = df[0].unique()
    x_sequence = []
    y_local = []
    ids = []
    for idx in id_list:
        n_row_idx = df[df[0] == idx].shape[0]
        if n_row_idx >= 15:
            ids.append(idx)
            x_sequence.append(df[df[0] == idx][feature_columns].iloc[:, 1:].values.tolist())
            y_local.append(df[df[0] == idx][41].iloc[:])        
    return x_sequence, y_local, ids

@tf.keras.utils.register_keras_serializable()
def F1_score(y_true, y_pred):
    precision = tf.keras.metrics.Precision()(y_true, y_pred)
    recall = tf.keras.metrics.Recall()(y_true, y_pred)

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1

max_len =50
embedding_dim = 39 

pickle_file_path = 'D:\IE7374_MLOps\Project_Datasets\Practice\Files\Training_set_A/sepsis_df.pkl'
with open(pickle_file_path, 'rb') as file:
    loaded_data = pickle.load(file)

with open('D:/IE7374_MLOps/Final_project/Early-Prediction-of-Sepsis/src/mlruns/0/bd39b548182e4137ac82ceb11834dfc0/artifacts/model/model.pkl', 'rb') as file:
    model = pickle.load(file)

train_df = loaded_data[loaded_data[0]<= 15000]
condition = (loaded_data.iloc[:, 0] > 15000) & (loaded_data.iloc[:, 0] <= 30000)
test_df = loaded_data[condition]

mean_values = train_df.mean()
train_df = train_df.fillna(mean_values)
test_df = test_df.fillna(mean_values)
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

scaler = MinMaxScaler()
scaler.fit(train_df.iloc[:, 1:-1])
train_df.iloc[:, 1:-1] = scaler.transform(train_df.iloc[:, 1:-1])
test_df.iloc[:, 1:-1] = scaler.transform(test_df.iloc[:, 1:-1])

train_df = train_df.groupby(train_df[0], group_keys=False).apply(delete_last_5_records)
train_df = train_df.groupby(train_df[0], group_keys=False).apply(select_last_50_or_all)
test_df = test_df.groupby(test_df[0], group_keys=False).apply(delete_last_5_records)
test_df = test_df.groupby(test_df[0], group_keys=False).apply(select_last_50_or_all)

feature_columns = test_df.iloc[:, :-1].columns

X_train, y_train, id_ls_train = create_sequence_tr_test_data(train_df, feature_columns)
X_test, y_test, id_ls_test = create_sequence_tr_test_data(test_df, feature_columns)

batch_size = 32
embedding_dim = 39

X_train = pad_trunc(X_train, max_len)
X_test = pad_trunc(X_test, max_len)

y_train = pad_sequences(y_train, padding='post', dtype='float32')
y_test = pad_sequences(y_test, padding='post', dtype='float32')  

X_train = np.reshape(X_train, (len(X_train), max_len, embedding_dim))
X_test = np.reshape(X_test, (len(X_test), max_len, embedding_dim))
y_train = np.array(y_train)
y_test = np.array(y_test)

test_predict = model.predict(X_test)
train_predict = model.predict(X_train)

thresholds = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
for threshold in thresholds:
    binary_predictions_train = (train_predict >= threshold).astype(int)
    binary_predictions_test = (test_predict >= threshold).astype(int)

    y_train_flat = np.ravel(y_train)
    binary_predictions_train_flat = np.ravel(binary_predictions_train)
    y_test_flat = np.ravel(y_test)
    binary_predictions_test_flat = np.ravel(binary_predictions_test)

    train_sensitivity = tf.keras.metrics.Recall()(y_train_flat, binary_predictions_train_flat)
    test_sensitivity = tf.keras.metrics.Recall()(y_test_flat, binary_predictions_test_flat)
    train_f1_score = f1_score(y_train_flat, binary_predictions_train_flat)
    test_f1_score =  f1_score(y_test_flat, binary_predictions_test_flat)

    print("threshold: ", threshold)
    print("train_sensitivity:", train_sensitivity)
    print("test_sensitivity:", test_sensitivity)
    print("test_f1_score", test_f1_score)
    print("train_f1_score", train_f1_score)
    print("train_cm", confusion_matrix(y_train_flat, binary_predictions_train_flat))
    print("test_cm", confusion_matrix(y_test_flat, binary_predictions_test_flat))
    print("\n")


