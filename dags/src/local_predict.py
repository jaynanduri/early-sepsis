from flask import Flask, request, jsonify
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
 
from flask import Flask, jsonify, request
import joblib
import os
import json
import pickle
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope
 
 
app = Flask(__name__)
def F1_score(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1

# Create Precision and Recall objects outside of the function
precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()


#model = load_model("D:/IE7374_MLOps/MLOps Project/Early_Prediction_of_Sepsis/dags/src/models/version_20231213161731/model.h5")



def fetch_latest_model(base_model_folder):
    base_model_folder_path = Path(base_model_folder)

    # Ensure the base model folder path exists
    if not base_model_folder_path.is_dir():
        raise FileNotFoundError(f"The specified base model folder '{base_model_folder}' does not exist.")

    # Get a list of all subdirectories (model versions) in the base model folder
    model_version_folders = [folder for folder in base_model_folder_path.iterdir() if folder.is_dir()]

    # Sort the model version folders based on the version in the folder name
    sorted_model_version_folders = sorted(model_version_folders, key=lambda x: x.name, reverse=True)

    if not sorted_model_version_folders:
        raise FileNotFoundError(f"No model versions found in the specified base model folder '{base_model_folder}'.")

    latest_model_version_folder = sorted_model_version_folders[0]

    model_files = list(latest_model_version_folder.glob("*.h5"))

    if not model_files:
        raise FileNotFoundError(f"No model files found in the latest version folder '{latest_model_version_folder}'.")

    sorted_model_files = sorted(model_files, key=lambda x: x.stem, reverse=True)
    latest_model_file_path = sorted_model_files[0]
    return str(latest_model_file_path)

# Example usage:
base_model_folder_path = "/Users/pavithra/Desktop/xyz2/dags/src/models"
latest_model_path = fetch_latest_model(base_model_folder_path)

# Register the custom metric function
with custom_object_scope({'F1_score': F1_score}):
    model = load_model(latest_model_path)

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
    id_list = df['id'].unique()
    x_sequence = []
    y_local = []
    ids = []
    for idx in id_list:
        n_row_idx = df[df['id'] == idx].shape[0]
        if n_row_idx >= 15:
            ids.append(idx)
            x_sequence.append(df[df['id'] == idx][feature_columns].iloc[:, 1:].values.tolist())
            y_local.append(df[df['id'] == idx]['SepsisLabel'].iloc[:])        
    return x_sequence, y_local, ids

 
 
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint that returns the status of the server.
    Returns:
        Response: A Flask response with status 200 and "healthy" as the body.
    """
    return {"status": "healthy"}
 
@app.route('/predict', methods=['GET'])
def predict():
    """
    Prediction route that normalizes input data, and returns model predictions.
    Returns:
        Response: A Flask response containing JSON-formatted predictions.
    """
    
    predict_df=pd.read_csv("/Users/pavithra/Desktop/xyz2/valid.csv")
 
    predict_df = predict_df.drop(predict_df.columns[[37,38]], axis=1)
    column_to_encode = 'Gender'
    encoded_columns = pd.get_dummies(predict_df[column_to_encode], prefix=column_to_encode)
    position = predict_df.columns.get_loc(column_to_encode)
    predict_df.drop(column_to_encode, axis=1, inplace=True)
    predict_df = pd.concat([predict_df.iloc[:, :position], encoded_columns, predict_df.iloc[:, position:]], axis=1)
    predict_df = predict_df.groupby('id').apply(lambda x: x.ffill()).reset_index(drop=True)
    predict_df = predict_df.groupby(predict_df['id'], group_keys=False).apply(delete_last_5_records)
    predict_df = predict_df.groupby(predict_df['id'], group_keys=False).apply(select_last_50_or_all)
 
    #mean_values = predict_df.mean()

    pickle_file_path = "/Users/pavithra/Desktop/xyz2/dags/src/mean_values.pkl"
    with open(pickle_file_path, 'rb') as file:
        mean_values = pickle.load(file)
    predict_df = predict_df.fillna(mean_values)
    predict_df = predict_df.fillna(0)
    path_to_scaler_pickle = "/Users/pavithra/Desktop/xyz2/dags/src/scaler.pkl"
    with open(path_to_scaler_pickle, 'rb') as file:
        scaler = pickle.load(file)
    scaler.fit(predict_df.iloc[:, 1:-1])
    predict_df.iloc[:, 1:-1] = scaler.transform(predict_df.iloc[:, 1:-1])
    
    feature_columns = predict_df.iloc[:, :-1].columns
    X_predict, y_predict, id_ls_predict = create_sequence_tr_test_data(predict_df, feature_columns)
 
    batch_size = 32
    embedding_dim = 39
    max_len = 50
 
    X_predict = pad_trunc(X_predict, max_len)
    y_predict = pad_sequences(y_predict, padding='pre', dtype='float32')  
    X_predict = np.reshape(X_predict, (len(X_predict), max_len, embedding_dim))
    y_predict = np.array(y_predict)
 
    predict_predict = model.predict(X_predict)
    binary_predictions_predict = (predict_predict >= 0.05).astype(int)
    max_pred_predict = np.max(binary_predictions_predict, axis = 1)
 
    prediction = max_pred_predict.tolist()
    #output = {'predictions': [{'result': pred} for pred in prediction]}
    ids = [int(identifier) for identifier in id_ls_predict]
    output = {'predictions': [{'id': identifier, 'result': pred} for identifier, pred in zip(ids, prediction)]}
    return jsonify(output)
 
 
if __name__ == '__main__':
    app.run(debug=True)
 