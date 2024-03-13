
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


'''
def feature_selection():

    train_files = []
    path = "/Users/pavithra/Documents/Projects/Early-Prediction-of-Sepsis/dags/data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setA"
    i = 0
    for f in os.listdir(path):
        if i>10000:
            break
        if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            train_files.append(f)
            i=i+1

    train_df = pd.DataFrame()
    path = "/Users/pavithra/Documents/Projects/Early-Prediction-of-Sepsis/dags/data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setA"
    for filename in os.listdir(path):
        if filename.endswith('.psv') and filename in train_files:
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
            file_id = os.path.splitext(filename)[0]
            file_id=int(file_id[1:])
            data['id'] = int(file_id)
            train_df = pd.concat([train_df, data], ignore_index=True)
    

    tr_last_column = train_df.pop(train_df.columns[-1])  # Remove the last column
    train_df.insert(0, tr_last_column.name, tr_last_column)

    print(train_df.columns)

    train_df = train_df.groupby('id').filter(lambda x: len(x) >=15 )
    train_df = train_df.groupby('id').fillna(method='ffill')
    train_df = train_df.fillna(0)
    print(train_df.columns)


    X=train_df.iloc[:, :-1]
    y=train_df.iloc[:, -1]

    train_set, valid_set, train_labels, valid_labels = train_test_split(X, y, test_size=0.33, random_state=42)

    rf_clf = RandomForestClassifier()
    rf_clf.fit(train_set, train_labels)

    predictions = rf_clf.predict(valid_set)



    feature_impt = pd.DataFrame(rf_clf.feature_importances_, index=train_set.columns)

    impt_features = feature_impt.loc[feature_impt[0] > 0.005]
    impt_features_list = list(impt_features.index.values)

    non_impt_features = feature_impt.loc[feature_impt[0] <= 0.005]
    non_impt_features_list = list(non_impt_features.index.values)


    print(impt_features_list)
    print(non_impt_features_list)

    imp_column_indices = [train_df.columns.get_loc(col) for col in impt_features_list]
    non_impt_column_indices = [train_df.columns.get_loc(col) for col in non_impt_features_list]

    print(imp_column_indices)
    print(non_impt_column_indices)

feature_selection()


def load():
    files = []
    path = os.path.join(os.path.dirname(__file__), '../data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setA')
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    random.shuffle(files)

    n_files = len(files)
    n_train = n_files * 6 // 10
    n_test = n_files * 2 // 10

    train_files = files[:n_train]
    test_files = files[n_train:n_train + n_test]
    valid_files = files[n_train + n_test:]


    train_df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith('.psv') and filename in train_files:
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path, sep='|') # Assuming the files are pipe-separated
            file_id = os.path.splitext(filename)[0]
            file_id=int(file_id[1:])
            data['id'] = file_id
            train_df = pd.concat([train_df, data], ignore_index=True)

    tr_last_column = train_df.pop(train_df.columns[-1])  # Remove the last column
    train_df.insert(0, tr_last_column.name, tr_last_column)

    test_df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith('.psv') and filename in test_files:
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
            file_id = os.path.splitext(filename)[0]
            file_id=file_id[1:]
            data['id'] = file_id
            test_df = pd.concat([test_df, data], ignore_index=True)

    te_last_column = test_df.pop(test_df.columns[-1])  # Remove the last column
    test_df.insert(0, te_last_column.name, te_last_column)

    valid_df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith('.psv') and filename in valid_files:
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
            file_id = os.path.splitext(filename)[0]
            file_id=file_id[1:]
            data['id'] = file_id
            valid_df = pd.concat([valid_df, data], ignore_index=True)
    
    va_last_column = valid_df.pop(valid_df.columns[-1])  # Remove the last column
    valid_df.insert(0, va_last_column.name, va_last_column)
    print(train_df.head(5))
    return train_df,valid_df,test_df

#a,b,c=load()


def feature(train_df,valid_df,test_df):

    train_df = train_df.groupby('id').filter(lambda x: len(x) >= 15)
    train_df = train_df.groupby('id').ffill()
    
    train_df = train_df.drop(train_df.columns[[8, 21, 28, 33]], axis=1)
    
    valid_df= valid_df.groupby('id').ffill()
    valid_df = valid_df.drop(valid_df.columns[[8, 21, 28, 33]], axis=1)

    test_df= test_df.groupby('id').ffill()
    test_df = test_df.drop(test_df.columns[[8, 21, 28, 33]], axis=1)

    return train_df,valid_df,test_df


#a,b,c= feature(a,b,c)

def preprocess(train_df,valid_df,test_df):
    
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    valid_df = valid_df.fillna(0)

    scaler = MinMaxScaler()
    train_df.iloc[:,1:-1] = scaler.fit_transform(train_df.iloc[:,1:-1])
    test_df.iloc[:,1:-1] = scaler.transform(test_df.iloc[:,1:-1])
    valid_df.iloc[:,1:-1] = scaler.transform(valid_df.iloc[:, 1:-1])

    
    file_path = os.path.join(os.path.dirname(__file__), '../data/','my_train_data_zero.csv')
    train_df.to_csv(file_path, index=False)
    file_path = os.path.join(os.path.dirname(__file__), '../data/','my_test_data_zero.csv')
    test_df.to_csv(file_path, index=False)
    file_path = os.path.join(os.path.dirname(__file__), '../data/','my_valid_data_zero.csv')
    valid_df.to_csv(file_path, index=False)

    return train_df,valid_df,test_df

#a,b,c=preprocess(a,b,c)

def sspreprocess(train_df,valid_df,test_df):
   
    mean_values = train_df.mean()
    train_df.fillna(mean_values, inplace=True)
    test_df.fillna(mean_values, inplace=True)
    valid_df.fillna(mean_values, inplace=True)
    
    scaler = MinMaxScaler()
    train_df.iloc[:,1:-1] = scaler.fit_transform(train_df.iloc[:,1:-1])
    test_df.iloc[:,1:-1] = scaler.transform(test_df.iloc[:,1:-1])
    valid_df.iloc[:,1:-1] = scaler.transform(valid_df.iloc[:, 1:-1])

    file_path = os.path.join(os.path.dirname(__file__), '../data/','my_train_data_mean.csv')
    train_df.to_csv(file_path, index=False)
    file_path = os.path.join(os.path.dirname(__file__), '../data/','my_test_data_mean.csv')
    test_df.to_csv(file_path, index=False)
    file_path = os.path.join(os.path.dirname(__file__), '../data/','my_valid_data_mean.csv')
    valid_df.to_csv(file_path, index=False)

    return train_df,valid_df,test_df

#a,b,c=sspreprocess(a,b,c)





'''


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import numpy as np
from collections import Counter
import random
import shutil
from sklearn.model_selection import train_test_split

def load_train_files():

        i=0
        train_files = []
        path = '/Users/pavithra/Desktop/xyz/dags/data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setA'
        for f in os.listdir(path):
            if i>20000:
                break
            if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
                train_files.append(f)
                i=i+1

        train_df = pd.DataFrame()
        path = '/Users/pavithra/Desktop/xyz/dags/data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setA'
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
        print(train_df.columns)
        return train_df

a=load_train_files()

def load_test_file():
        i=0
        test_files = []
        path = '/Users/pavithra/Desktop/xyz/dags/data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setB'
        for f in os.listdir(path):
            if i>10000:
                break
            if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
                test_files.append(f)
                i=i+1

        test_df = pd.DataFrame()
        path = '/Users/pavithra/Desktop/xyz/dags/data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setB'
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

        return test_df

b=load_test_file()


def feature_engineering(train_df,test_df):
    
    train_df = train_df.drop(train_df.columns[[37,38]], axis=1)
    column_index_to_encode = 36
    encoded_columns = pd.get_dummies(train_df.iloc[:, column_index_to_encode], prefix='column_36')
    train_df.drop(train_df.columns[column_index_to_encode], axis=1, inplace=True)
    train_df = pd.concat([train_df, encoded_columns], axis=1)
    train_df = train_df.drop('EtCO2', axis=1)

    test_df = test_df.drop(test_df.columns[[37,38]], axis=1)
    encoded_columns = pd.get_dummies(test_df.iloc[:, column_index_to_encode], prefix='column_36')
    test_df.drop(test_df.columns[column_index_to_encode], axis=1, inplace=True)
    test_df = pd.concat([test_df, encoded_columns], axis=1)

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

    mean_values.to_csv('mean_values_.csv')

    
    train_df.fillna(mean_values, inplace=True)
    test_df.fillna(mean_values, inplace=True)

    train_df.to_pickle('meantrain_df.pkl')
    test_df.to_pickle('meantest_df.pkl')

    X=train_df.iloc[:, :-1]
    y=train_df.iloc[:, -1]

    train_set, valid_set, train_labels, valid_labels = train_test_split(X, y, test_size=0.33, random_state=42)

    rf_clf = RandomForestClassifier()
    rf_clf.fit(train_set, train_labels)

    predictions = rf_clf.predict(valid_set)
    feature_impt = pd.DataFrame(rf_clf.feature_importances_, index=train_set.columns)
    non_impt_features = feature_impt.loc[feature_impt[0] <= 0.005]
    non_impt_features_list = list(non_impt_features.index.values)
    #non_impt_column_indices = [train_df.columns.get_loc(col) for col in non_impt_features_list]

    print(non_impt_features_list)
    print(feature_impt)
    train_df = train_df.drop(non_impt_features_list, axis=1)
    test_df = test_df.drop(non_impt_features_list, axis=1)

    train_df.to_pickle('ttrain_df.pkl')
    test_df.to_pickle('ttest_df.pkl')


feature_engineering(a,b)

