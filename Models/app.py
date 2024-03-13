from flask import Flask, request, jsonify
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


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


app = Flask(__name__)
model = load_model('/Users/pavithra/Downloads/model.h5')


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
    predict_df=pd.read_csv('/Users/pavithra/Desktop/xyz2/valid.csv')

    predict_df = predict_df.drop(predict_df.columns[[37,38]], axis=1)
    column_to_encode = 'Gender'
    encoded_columns = pd.get_dummies(predict_df[column_to_encode], prefix=column_to_encode) 
    position = predict_df.columns.get_loc(column_to_encode) 
    predict_df.drop(column_to_encode, axis=1, inplace=True) 
    predict_df = pd.concat([predict_df.iloc[:, :position], encoded_columns, predict_df.iloc[:, position:]], axis=1)
    predict_df = predict_df.groupby('id').apply(lambda x: x.ffill()).reset_index(drop=True)
    predict_df = predict_df.groupby(predict_df['id'], group_keys=False).apply(delete_last_5_records)
    predict_df = predict_df.groupby(predict_df['id'], group_keys=False).apply(select_last_50_or_all)

    mean_values = predict_df.mean()
    predict_df = predict_df.fillna(mean_values)
    predict_df = predict_df.fillna(0)
    scaler = MinMaxScaler()
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

    ## import model

    predict_predict = model.predict(X_predict)
    binary_predictions_predict = (predict_predict >= 0.005).astype(int)
    max_pred_predict = np.max(binary_predictions_predict, axis = 1)
    #binary_predictions_predict_flat = np.ravel(binary_predictions_predict)

    # Make predictions with the model
    prediction = max_pred_predict.tolist()
    output = {'predictions': [{'result': pred} for pred in prediction]}
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
