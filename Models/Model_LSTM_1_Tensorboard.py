## ffill + mean imputation normalized data + LSTM



import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np, os, sys
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
import sys
import warnings
from urllib.parse import urlparse
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from packaging import version
tf.debugging.experimental.enable_dump_debug_info('./logs/', tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
from keras import backend as K


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


@tf.keras.utils.register_keras_serializable()
def F1_score(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1

# Create Precision and Recall objects outside of the function
precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    max_len = 50
    embedding_dim = 39 

    pickle_file_path = '/Users/pavithra/Desktop/xyz 2/sepsis_df_mean_imput_train_test.pkl'

    with open(pickle_file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    # Access the loaded arrays using the keys
    X_train = loaded_data['X_train_mean_imputed']
    y_train = loaded_data['y_train_mean_imputed']
    X_test = loaded_data['X_test_mean_imputed']
    y_test = loaded_data['y_test_mean_imputed']

    ### define hyperparameters 
    num_neurons = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    drop_rate = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    optimizer = sys.argv[5] if len(sys.argv) > 5 else 'RMSprop'
    
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    with mlflow.start_run():
        tf.keras.backend.set_image_data_format("channels_last")
        model = Sequential()
        model.add(LSTM(num_neurons, input_shape=(max_len, embedding_dim), return_sequences=True))
        model.add(Dropout(drop_rate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.Recall(), F1_score])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])


        test_predict = model.predict(X_test)
        train_predict = model.predict(X_train)

        binary_predictions_train = (train_predict >= 0.5).astype(int)
        binary_predictions_test = (test_predict >= 0.5).astype(int)

        y_train_flat = np.ravel(y_train)
        binary_predictions_train_flat = np.ravel(binary_predictions_train)
        y_test_flat = np.ravel(y_test)
        binary_predictions_test_flat = np.ravel(binary_predictions_test)

        train_sensitivity = tf.keras.metrics.Recall()(y_train_flat, binary_predictions_train_flat)
        test_sensitivity = tf.keras.metrics.Recall()(y_test_flat, binary_predictions_test_flat)
        train_f1_score = f1_score(y_train_flat, binary_predictions_train_flat)
        test_f1_score =  f1_score(y_test_flat, binary_predictions_test_flat)

        print("train_sensitivity:", train_sensitivity)
        print("test_sensitivity:", test_sensitivity)
        print("test_f1_score", test_f1_score)
        print("train_f1_score", train_f1_score)
        print("train_cm", confusion_matrix(y_train_flat, binary_predictions_train_flat))
        print("test_cm", confusion_matrix(y_test_flat, binary_predictions_test_flat))

        mlflow.log_param("num_neurons", num_neurons)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("drop_rate", drop_rate)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_metric("train_sensitivity", train_sensitivity)
        mlflow.log_metric("test_sensitivity", test_sensitivity)
        mlflow.log_metric("train_f1_score", train_f1_score)
        mlflow.log_metric("test_f1_score", test_f1_score)

        predictions = model.predict(X_train)
        signature = infer_signature(X_train, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                model, "model", registered_model_name="SepsisModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)



## tensorboard --logdir logs



