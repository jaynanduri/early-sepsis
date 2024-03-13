## ffill + mean imputation normalized data + LSTM
import os
import time 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np, os, sys
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
import sys
import warnings
from urllib.parse import urlparse
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.preprocessing import MinMaxScaler

def delete_last_5_records(group):
    return group.iloc[:-5]


def select_last_50_or_all(group):
    return group.tail(50) if len(group) >= 50 else group


def pad_trunc(data, max_len):
    new_data = []
    zero_vec = []
    for _ in range(len(data[0][0])):
        zero_vec.append(0.0)
    
    for sample in data:
        if len(sample) > max_len:  # truncate
            temp = sample[:max_len]
        elif len(sample) < max_len:
            temp = sample
            number_additional_elements = max_len - len(sample)
            for _ in range(number_additional_elements):
                temp.insert(0, zero_vec)  # insert zero vectors at the beginning
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


def F1_score(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1

# Create Precision and Recall objects outside of the function
precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()


def create_versioned_directory(base_folder, version_prefix='version'):
    timestamp = time.strftime("%Y%m%d%H%M%S")
    versioned_folder = os.path.join(base_folder, f'{version_prefix}_{timestamp}')

    if not os.path.exists(versioned_folder):
        os.makedirs(versioned_folder)

    print(f'Created versioned directory: {versioned_folder}')
    return versioned_folder


def save_model_with_version(model, versioned_folder):
    model_filename = os.path.join(versioned_folder, 'model.h5')
    model.save(model_filename)
    print(f'Model saved in versioned directory: {versioned_folder}')


def model(X_train,X_test,y_train,y_test, model_version_folder):

        max_len = 50
        embedding_dim = 39
        num_neurons = 50
        epochs = 60
        batch_size = 64
        drop_rate = 0.2
        optimizer = 'RMSprop'


        model = Sequential()
        model.add(LSTM(num_neurons, input_shape=(max_len, embedding_dim), return_sequences=True))
        model.add(Dropout(drop_rate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.Recall(), F1_score])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

        # Make predictions
        test_predict = model.predict(X_test)
        train_predict = model.predict(X_train)

        save_model_with_version(model, model_version_folder)

pickle_file_path = '/Users/pavithra/Desktop/xyz2/sepsis_df_mean_imput_train_test.pkl'
with open(pickle_file_path, 'rb') as file:
    loaded_data = pickle.load(file)

# Access the loaded arrays using the keys
X_train = loaded_data['X_train_mean_imputed']
y_train = loaded_data['y_train_mean_imputed']
X_test = loaded_data['X_test_mean_imputed']
y_test = loaded_data['y_test_mean_imputed']

base_folder = "/Users/pavithra/Desktop/xyz2/dags/src/models"
versioned_folder = create_versioned_directory(base_folder)
model(X_train, X_test, y_train, y_test, versioned_folder)