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

def load_train_files(**kwargs):
        i=0
        train_files = []
        path = os.path.join(os.path.dirname(__file__), '../data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setA')
        for f in os.listdir(path):
            if i>18000:
                break
            if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
                train_files.append(f)
                i=i+1

        train_df = pd.DataFrame()
        path = os.path.join(os.path.dirname(__file__), '../data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setA')
        for filename in os.listdir(path):
            if filename.endswith('.psv') and filename in train_files:
                file_path = os.path.join(path, filename)
                data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
                file_id = os.path.splitext(filename)[0]
                file_id=int(file_id[1:])
                data['id'] = file_id
                train_df = pd.concat([train_df, data], ignore_index=True)

        tr_last_column = train_df.pop(train_df.columns[-1])  # Remove the last column
        train_df.insert(0, tr_last_column.name, tr_last_column)
        train_serialized_data = pickle.dumps(train_df)
        ti = kwargs['ti']
        ti.xcom_push(key='train_data', value=train_serialized_data)


def load_test_files(**kwargs):

        i=0
        test_files = []
        path = os.path.join(os.path.dirname(__file__), '../data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setB')
        for f in os.listdir(path):
            if i>10100:
                break
            if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
                test_files.append(f)
                i=i+1

        test_df = pd.DataFrame()
        path = os.path.join(os.path.dirname(__file__), '../data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setB')
        for filename in os.listdir(path):
            if filename.endswith('.psv') and filename in test_files:
                file_path = os.path.join(path, filename)
                data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
                file_id = os.path.splitext(filename)[0]
                file_id=int(file_id[1:])
                data['id'] = file_id
                test_df = pd.concat([test_df, data], ignore_index=True)

        ts_last_column = test_df.pop(test_df.columns[-1])  # Remove the last column
        test_df.insert(0, ts_last_column.name, ts_last_column)
        test_serialized_data = pickle.dumps(test_df)
        ti = kwargs['ti']
        ti.xcom_push(key='test_data', value=test_serialized_data) 

def feature_engineering(**kwargs):
    ti = kwargs['ti']
    train_data = ti.xcom_pull(task_ids='load_train_files', key='train_data')
    test_data = ti.xcom_pull(task_ids='load_test_files', key='test_data')

    train_df = pickle.loads(train_data)

    train_df = train_df.drop(train_df.columns[[37,38]], axis=1)
    column_to_encode = 'Gender'
    encoded_columns = pd.get_dummies(train_df[column_to_encode], prefix=column_to_encode) 
    position = train_df.columns.get_loc(column_to_encode) 
    train_df.drop(column_to_encode, axis=1, inplace=True) 
    train_df = pd.concat([train_df.iloc[:, :position], encoded_columns, train_df.iloc[:, position:]], axis=1)

    test_df = pickle.loads(test_data)
    test_df = test_df.drop(test_df.columns[[37,38]], axis=1)
    encoded_columns = pd.get_dummies(test_df[column_to_encode], prefix=column_to_encode) 
    position = test_df.columns.get_loc(column_to_encode) 
    test_df.drop(column_to_encode, axis=1, inplace=True) 
    test_df = pd.concat([test_df.iloc[:, :position], encoded_columns, test_df.iloc[:, position:]], axis=1)

    train_df = train_df.groupby('id').apply(lambda x: x.ffill()).reset_index(drop=True)
    test_df = test_df.groupby('id').apply(lambda x: x.ffill()).reset_index(drop=True)

    train_serialized_data = pickle.dumps(train_df)
    test_serialized_data = pickle.dumps(test_df)

    ti = kwargs['ti']
    ti.xcom_push(key='f_train_data', value=train_serialized_data)
    ti.xcom_push(key='f_test_data', value=test_serialized_data)

def mean_imputation(**kwargs):
    ti = kwargs['ti']
    train_data = ti.xcom_pull(task_ids='feature_engineering', key='f_train_data')
    test_data = ti.xcom_pull(task_ids='feature_engineering', key='f_test_data')
    train_df = pickle.loads(train_data)
    test_df = pickle.loads(test_data)

    def delete_last_5_records(group):
        return group.iloc[:-5]

    def select_last_50_or_all(group):
        return group.tail(50) if len(group) >= 50 else group

    train_df = train_df.groupby(train_df['id'], group_keys=False).apply(delete_last_5_records)
    train_df = train_df.groupby(train_df['id'], group_keys=False).apply(select_last_50_or_all)
    test_df = test_df.groupby(test_df['id'], group_keys=False).apply(delete_last_5_records)
    test_df = test_df.groupby(test_df['id'], group_keys=False).apply(select_last_50_or_all)

    mean_values = train_df.mean()
    mean_values.to_csv('mean.csv')
    train_df = train_df.fillna(mean_values)
    test_df = test_df.fillna(mean_values)
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    scaler = MinMaxScaler()
    scaler.fit(train_df.iloc[:, 1:-1])
    train_df.iloc[:, 1:-1] = scaler.transform(train_df.iloc[:, 1:-1])
    test_df.iloc[:, 1:-1] = scaler.transform(test_df.iloc[:, 1:-1])

    train_serialized_data = pickle.dumps(train_df)
    test_serialized_data = pickle.dumps(test_df)

    ti = kwargs['ti']
    ti.xcom_push(key='mean_train_data', value=train_serialized_data)
    ti.xcom_push(key='mean_test_data', value=test_serialized_data)

def training(**kwargs):

    ti = kwargs['ti']
    train_data = ti.xcom_pull(task_ids='mean_imputation', key='mean_train_data')
    test_data = ti.xcom_pull(task_ids='mean_imputation', key='mean_test_data')
    train_df = pickle.loads(train_data)
    test_df = pickle.loads(test_data)

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import tensorflow as tf

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
    
    feature_columns = test_df.iloc[:, :-1].columns

    X_train, y_train, id_ls_train = create_sequence_tr_test_data(train_df, feature_columns)
    X_test, y_test, id_ls_test = create_sequence_tr_test_data(test_df, feature_columns)

    batch_size = 32
    embedding_dim = 39
    max_len = 50
    embedding_dim = 39

    X_train = pad_trunc(X_train, max_len)
    X_test = pad_trunc(X_test, max_len)

    y_train = pad_sequences(y_train, padding='post', dtype='float32')
    y_test = pad_sequences(y_test, padding='post', dtype='float32')  

    X_train = np.reshape(X_train, (len(X_train), max_len, embedding_dim))
    X_test = np.reshape(X_test, (len(X_test), max_len, embedding_dim))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train_serialized_data = pickle.dumps(X_train)
    X_test_serialized_data = pickle.dumps(X_test)
    y_train_serialized_data = pickle.dumps(y_train)
    y_test_serialized_data = pickle.dumps(y_test)

    ti = kwargs['ti']
    ti.xcom_push(key='x_train_data', value=X_train_serialized_data)
    ti.xcom_push(key='x_test_data', value=X_test_serialized_data)
    ti.xcom_push(key='y_train_data', value=y_train_serialized_data)
    ti.xcom_push(key='y_test_data', value=y_test_serialized_data)


def model(**kwargs):

    ti = kwargs['ti']
    X_train_data = ti.xcom_pull(task_ids='training', key='x_train_data')
    X_test_data = ti.xcom_pull(task_ids='training', key='x_test_data')
    y_train_data = ti.xcom_pull(task_ids='training', key='y_train_data')
    y_test_data = ti.xcom_pull(task_ids='training', key='y_test_data')
    X_train= pickle.loads(X_train_data)
    X_test = pickle.loads(X_test_data)
    y_train = pickle.loads(y_train_data)
    y_test = pickle.loads(y_test_data)

    #import tensorflow as tf
    #from tensorflow.keras.preprocessing.sequence import pad_sequences
    

    if __name__ == "__main__":
        warnings.filterwarnings("ignore")
    
        max_len = 50
        embedding_dim = 39
    
        ### define hyperparameters
        num_neurons = int(sys.argv[1]) if len(sys.argv) > 1 else 50
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 64
        drop_rate = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
        optimizer = sys.argv[5] if len(sys.argv) > 5 else 'RMSprop'
    
    
        with mlflow.start_run():
            tf.keras.backend.set_image_data_format("channels_last")
    
            model = Sequential()
            model.add(LSTM(num_neurons, input_shape=(max_len, embedding_dim), return_sequences=True))
            model.add(Dropout(drop_rate))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.Recall(), F1_score])
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    
    
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
    
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="SepsisModel", signature=signature
                )
            else:
                mlflow.sklearn.log_model(model, "model", signature=signature)
 