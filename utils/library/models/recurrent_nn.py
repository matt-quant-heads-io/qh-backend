import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU, Dropout
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model
import numpy as np

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
import pandas as pd

from config.config import Config
from utils.utils import create_train_test_set, create_pred_data


CONFIG = Config()
MODEL_CONFIG = CONFIG[CONFIG["parent_key"]]


class RecurrentNNWrapper(object):

    _model_name_ = "recurrent_nn"

    def __init__(self, model_details=None, verbose=False):
        self._verbose = verbose
        self._model_details = model_details
        self._model = None
        self._is_compiled = False

    def is_model_compiled(self):
        try:
            return self._is_compiled
        except Exception as e:
            msg = "Error in is_model_compiled"
            raise e(msg)

    # TODO: add this as virutal function in model base class from which all models will inherit and have to implement
    def build_model(self, verbose=False):
        try:
            if self._model_details is None:
                msg = f"None value detected for build_instructions_dict dict while attempting" \
                      f" to instantiate {self.__class__.__name__}"
                raise ValueError(msg)
            if self._model_details["load_model_from_file"]:
                model = self.load_model(model_path=self._model_details["persisted_model_path"], verbose=True)
                optimizer = self._model_details["model_config"]["optimizer"]
                loss = self._model_details["model_config"]["loss"]
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
                self._is_compiled = True
                self._model = model
            else:
                params_dict = self._model_details["model_config"]
                model = Sequential()
                model.add(LSTM(units=params_dict["num_units"],
                               activation=params_dict["activation_func"],
                               input_shape=(params_dict["num_inputs"], params_dict["num_features"])))
                model.add(LeakyReLU(alpha=params_dict["alpha"]))
                model.add(Dropout(params_dict["dropout"]))
                model.add(Dense(units=params_dict["output_units"]))
                model.compile(params_dict["optimizer"], params_dict["loss"], metrics=['accuracy'])
                self._is_compiled = True
                if verbose:
                    model.summary()
                self._model = model
        except Exception as e:
            msg = "Error in build_model"
            raise e(msg)

    def get_pred_data(self):
        try:
            pred_dir = self._model_details.get("predict_data_path")
            # TODO: make the training data file a config param
            df = pd.read_csv(f'{pred_dir}/pred_data_df.csv', header=0, index_col='datetime')
            num_features = self._model_details.get("model_config")["num_features"]
            scale_data = self._model_details.get("model_config")["scale_data"]
            num_inputs = self._model_details.get("model_config")["num_inputs"]
            X_train = create_pred_data(df=df, num_features=num_features, scale_data=scale_data, verbose=self._verbose)
            if len(X_train) == 0:
                msg = "X_train cannot be empty"
                raise ValueError(msg)
            return X_train[:num_inputs, :]
        except Exception as e:
            msg = "Error in get_training_data"
            raise e(msg)

    def get_training_data(self):
        try:
            train_dir = self._model_details.get("train_data_path")
            train_file = self._model_details.get("train_file")
            # TODO: make the training data file a config param
            df = pd.read_csv(f'{train_dir}/{train_file}', header=0, index_col='datetime')
            train_split_size = self._model_details.get("model_config")["train_split_size"]
            num_features = self._model_details.get("model_config")["num_features"]
            X_train, X_test, y_train, y_test, train_test_split = create_train_test_set(df=df, num_features=num_features,
                                                                                       train_split=train_split_size)
            return X_train, X_test, y_train, y_test, train_test_split
        except Exception as e:
            msg = "Error in get_training_data"
            raise e(msg)

    # TODO: add this to a model base class from which all models will inherit from
    def load_model(self, verbose=False):
        try:
            model_path = self._model_details.get('persisted_model_path')
            if model_path is None:
                msg = "model_path cannot be None type"
                raise Exception(msg)
            else:
                if verbose:
                    print(f"Loading model from {model_path}")
                model = keras.models.load_model(model_path, compile=False)
                optimizer = self._model_details["model_config"]["optimizer"]
                loss = self._model_details["model_config"]["loss"]
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
                self._model = model
        except Exception as e:
            msg = "Error in load_model"
            raise e(msg)

    # TODO: add this as virutal function in model base class from which all models will inherit and have to implement
    def train(self, training_data_tuple=None, verbose=False):
        try:
            training_params = self._model_details.get("model_config")
            if training_params is None:
                msg = f"model_config not in {self._model_details}"
                raise KeyError(msg)
            elif training_data_tuple is None:
                msg = "training_data_tuple cannot be None"
                raise ValueError(msg)
            else:
                X_train, X_test, y_train, y_test, train_test_split = training_data_tuple
                num_inputs = training_params.get("num_inputs")
                batch_size = training_params.get("batch_size")
                epochs = training_params.get("epochs")
                generator = TimeseriesGenerator(X_train, y_train, length=num_inputs, batch_size=batch_size)
                self._model.fit_generator(generator, epochs=epochs)
                persisted_model_path = self._model_details.get("persisted_model_path")
                self._model.save(persisted_model_path)
                if verbose:
                    print(f"Saved recurrent_nn model type to "
                          f"{persisted_model_path}")
        except Exception as e:
            msg = "Error in train"
            raise e(msg)

    # TODO: add this to a model base class from which all models will inherit from
    def predict(self, pred_data=None):
        try:
            if pred_data is None:
                msg = "pred_data cannot be None type"
                raise Exception(msg)
            else:
                pred_params_dict = self._model_details.get("model_config")
                generator_test = TimeseriesGenerator(pred_data,
                                                     pred_data,
                                                     length=pred_params_dict["num_inputs"]-1,
                                                     batch_size=1)
                pred_result = self._format_predict(self._model.predict(generator_test))
                return pred_result
        except Exception as e:
            msg = "Error in predict"
            raise e(msg)

    def _format_predict(self, pred_result=None):
        try:
            if pred_result is None:
                msg = "pred_result cannot be None type"
                raise Exception(msg)
            else:
                return {"model_name": self._model_name_, "prediction": list(pred_result.ravel())}
        except Exception as e:
            msg = "Error in _format_predict"
            raise e(msg)
