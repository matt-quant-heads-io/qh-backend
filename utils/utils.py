import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def convert_class_path_to_file_name(class_path=None, verbose=False):
    try:
        if class_path is None:
            msg = 'class_path cannot be None type'
            raise ValueError(msg)
        else:
            file_name = None
            if '.' in class_path:
                if verbose:
                    print(f'converting class_path {class_path}')
                file_name = class_path.split('.')[-1]
            return file_name
    except Exception as e:
        msg = "Error in convert_class_path_to_file_name"
        raise e(msg)


def preprocess_df(df=None, target_column=None, verbose=False):
    try:
        if df is None or len(df) == 0 or target_column is None:
            raise Exception("params df and target_column cannot be None type")
        else:
            y = df[[target_column]]
            df = df.drop(target_column, axis=1)
            df.reset_index(inplace=True)
            df = df.drop('datetime', axis=1)
            if verbose:
                print(f"y:\n{y.head(3)}\n")
                print(f"X:\n{df.head(3)}")
            return df, y
    except Exception as e:
        msg = "error in create_train_test_set"
        raise e(msg)


def create_pred_data(df=None, num_features=None, scale_data=False, verbose=False):
    """General function for formatting predict data"""
    try:
        if df is None or len(df) == 0:
            msg = "Params df cannot be None type"
            raise Exception(msg)
        if 'datetime' in df.columns:
            df = df.drop('datetime', axis=1)
        if verbose:
            print(f"predict data columns: {list(df.columns)}")
        if num_features is None:
            num_features = len(df.columns)
        X_train = np.array(df.iloc[:, :num_features])
        if scale_data:
            X_scaler = MinMaxScaler(feature_range=(-1, 1)) # scale so that all the X data will range from 0 to 1
            X_scaler.fit(X_train)
            X_train = X_scaler.transform(X_train)
        return X_train
    except Exception as e:
        msg = "error in create_train_test_set"
        raise e(msg)


def create_train_test_set(df=None, train_split=None, target_column='close', num_features=None, scale_data=False,
                          verbose=False):
    """General function for splitting data into Xtrain, Xtest, ytrain, ytest"""
    try:
        if df is None or train_split is None or len(df) == 0:
            msg = "Params df and train_split cannot be None type"
            raise Exception(msg)
        df, y = preprocess_df(df=df, target_column=target_column)
        if verbose:
            print(f"training data columns: {list(df.columns)}")
        if num_features is None:
            num_features = len(df.columns)
        # Split the data into training and testing
        train_test_split = int(len(df) * train_split)
        Xtrain = np.array(df.iloc[:train_test_split, :num_features])
        Xtest = np.array(df.iloc[train_test_split:, :num_features])
        ytrain = np.array(y.iloc[:train_test_split])
        ytest = np.array(y.iloc[train_test_split:])
        if scale_data:
            #Scale Xtrain, Xtest
            Xscaler = MinMaxScaler(feature_range=(-1, 1)) # scale so that all the X data will range from 0 to 1
            Xscaler.fit(Xtrain)
            Xtrain = Xscaler.transform(Xtrain)
            Xscaler.fit(Xtest)
            Xtest = Xscaler.transform(Xtest)
            #Scale ytrain, ytest
            Yscaler = MinMaxScaler(feature_range=(-1, 1))
            Yscaler.fit(ytrain)
            ytrain = Yscaler.transform(ytrain)
        return Xtrain, Xtest, ytrain, ytest, train_test_split
    except Exception as e:
        msg = "error in create_train_test_set"
        raise e(msg)



# TODO: refactor these helper functions for cnn model integration
# def bucket_to_dir(bucket_name):
#     return '_'.join(bucket_name.split('-')[2:])
#
#
def load_images(conn, bucket_path):
    for key in conn.list_objects(Bucket=bucket_path)['Contents']:
        file_name = key['Key']
        # TODO: change this to read in the target from Config
        target = "./"
        conn.download_file(bucket_path, file_name, target)