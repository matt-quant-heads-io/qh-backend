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
            # THIS IS RNN EXAMPLE FOR STARTING POINT
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
            pass
        except Exception as e:
            msg = "Error in get_training_data"
            raise e(msg)

    def get_training_data(self):
        try:
            pass
        except Exception as e:
            msg = "Error in get_training_data"
            raise e(msg)

    # TODO: add this to a model base class from which all models will inherit from
    def load_model(self, verbose=False):
        try:
            pass
        except Exception as e:
            msg = "Error in load_model"
            raise e(msg)

    # TODO: add this as virutal function in model base class from which all models will inherit and have to implement
    def train(self, training_data_tuple=None, verbose=False):
        try:
            pass
        except Exception as e:
            msg = "Error in train"
            raise e(msg)

    # TODO: add this to a model base class from which all models will inherit from
    def predict(self, pred_data=None):
        try:
            pass
        except Exception as e:
            msg = "Error in predict"
            raise e(msg)

    def _format_predict(self, pred_result=None):
        try:
            pass
        except Exception as e:
            msg = "Error in _format_predict"
            raise e(msg)
