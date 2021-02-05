# import os
# import argparse
# import json
# import datetime

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

import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates


CONFIG = Config()
MODEL_CONFIG = CONFIG[CONFIG["parent_key"]]
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model

from config.config import Config


class ConvolutionalNNWrapper(object):

    _model_name_ = "convolutional_nn"

    def __init__(self, model_details=None, verbose=False):
        self._verbose = verbose
        self._model_details = model_details
        self._model = None
        self._is_compiled = False

    def is_model_compiled(self):
        try:
            return self._is_compiled
        except Exception as e:
            msg = "Error in is_model_compiled()"
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
    def build_model(self, verbose=False):
        try:
            if self._model_details is None:
                msg = f"None value detected for build_instructions_dict dict while attempting" \
                      f" to instantiate {self.__class__.__name__}"
                raise ValueError(msg)
            optimizer = self._model_details["model_config"]["optimizer"]
            loss = self._model_details["model_config"]["loss"]
            if self._model_details["load_model_from_file"]:
                model = self.load_model(model_path=self._model_details["persisted_model_path"], verbose=True)
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
                self._is_compiled = True
                self._model = model
            else:
                params_dict = self._model_details["model_config"]
                num_class_labels = len(params_dict.get("patterns"))
                model = tf.keras.models.Sequential([
                    # This is the first convolution
                    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    # The second convolution
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    # The third convolution
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    # The fourth convolution
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    # Flatten the results to feed into a DNN
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dropout(0.5),
                    # 512 neuron hidden layer
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(num_class_labels, activation='softmax')
                ])
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
                self._is_compiled = True
                if verbose:
                    model.summary()
                self._model = model
        except Exception as e:
            msg = "Error in __build_model()"
            raise e(msg)

    def get_price_data(self, ticker="AAPL"):
        try:
            #TODO: Change this mechanism to work with polygon, integrate lookback period from config
            t = yf.Ticker(ticker)
            hist = t.history(period="120d")
            return hist
        except:
            msg = "Error in get_price_data"
            raise Exception(msg)

    def build_chart(self, abs_file_path=None):
        try:
            if abs_file_path is None:
                msg = "abs_file_path cannot be None"
                raise ValueError(msg)
            #TODO: remove x-axis and y-axis
            hist_df = self.get_price_data(self._model_details.get("symbol"))
            hist_df.reset_index(inplace=True)
            hist_df.rename(columns={"time": "Date"}, inplace=True)
            hist_df['Date'] = hist_df['Date'].map(mdates.date2num)
            ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
            ax1.xaxis_date()
            candlestick_ohlc(ax1, hist_df.values, width=2, colorup='g')
            pred_dir = self._model_details.get("predict_data_path")
            pred_file = self._model_details.get("pred_file")
            abs_file_path = f'{pred_dir}/{pred_file}'
            plt.savefig(f"{abs_file_path}")
        except:
            msg = "Error in build_chart"
            raise Exception(msg)

    def get_pred_data(self):
        try:
            pred_dir = self._model_details.get("predict_data_path")
            pred_file = self._model_details.get("pred_file")
            pred_abs_path = f'{pred_dir}/{pred_file}'
            # CREATE THE CHART
            self.build_chart(pred_abs_path)
            #TODO: Turn target_size into config.py params
            img = image.load_img(pred_abs_path, target_size=(300, 300))
            if img is None:
                msg = "img cannot be empty"
                raise ValueError(msg)
            x = image.img_to_array(img)
            img_array = np.expand_dims(x, axis=0)
            pred_data = np.vstack([img_array])
            return pred_data
        except Exception as e:
            msg = "Error in get_training_data"
            raise e(msg)

    def get_training_data(self):
        try:
            train_dir = self._model_details.get("train_data_path")
            # TODO: LOAD TRAINING IMAGES
            val_dir = self._model_details.get("test_data_path")
            # TODO: add validation_data_s3_bucket key in config.model_config.convolutional_nn_config
            training_datagen = ImageDataGenerator(rescale=1. / 255)
            validation_datagen = ImageDataGenerator(rescale=1. / 255)
            return train_dir, training_datagen, val_dir, validation_datagen
        except Exception as e:
            msg = "Error in get_training_data"
            raise e(msg)

    # TODO: add this as virutal function in model base class from which all models will inherit and have to implement
    def train(self, training_data_tuple=None, verbose=False):
        try:
            if training_data_tuple is None:
                msg = f"training_data_tuple cannot be None"
                raise ValueError(msg)
            else:
                train_dir, training_datagen, val_dir, validation_datagen = training_data_tuple
                train_generator = training_datagen.flow_from_directory(train_dir, target_size=(300, 300),
                                                                       class_mode='categorical', batch_size=1)
                validation_generator = validation_datagen.flow_from_directory(val_dir, target_size=(300, 300),
                                                                              class_mode='categorical',
                                                                              batch_size=1)

                self._model.fit(train_generator, epochs=1, steps_per_epoch=1,
                                validation_data=validation_generator, verbose=1, validation_steps=1)

                persisted_model_path = self._model_details.get("persisted_model_path")
                self._model.save(persisted_model_path)
                if verbose:
                    print(f"Saved convolutional_nn model type to path {persisted_model_path}")
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
                # TODO: PREDICT LOGIC GOES HERE
                # num_label = np.argmax(self._model.predict(pred_data), axis=-1)[0]
                # pred_result = self._model.predict(_image, batch_size=1)[0][num_label]
                # pred_result (label_prob, class_label)
                pred_result = self._format_predict(self._model.predict(pred_data))





                # TODO: CALL _format_predict HERE
                # pred_result = self._format_predict(self._model.predict(pred_data))
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
                # TODO: FORMAT PREDICTION LOGIC GOES HERE
                pred_result = np.array(pred_result[0])
                num_label = np.argmax(pred_result)
                pattern_prob = pred_result[num_label]
                pattern_label = self._model_details.get('model_config')['patterns'][num_label]
                return {"model_name": self._model_name_, "prediction": (pattern_label, pattern_prob)}
        except Exception as e:
            msg = "Error in _format_predict"
            raise e(msg)







