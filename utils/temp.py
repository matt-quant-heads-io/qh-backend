from config.config import Config
from library.feeds.polygon_feed import PolygonFeed
from library.models.recurrent_nn import RecurrentNNWrapper
from library.models.convolutional_nn import ConvolutionalNNWrapper
from library.models.pipeline import Pipeline
from utils.utils import create_train_test_set

import argparse
import datetime as dt
from typing import List
import time
import ast
import pandas as pd
from tensorflow import keras
from keras.models import load_model
#
import tensorflow.keras.backend as K
K.clear_session()
pipeline = Pipeline(verbose=True)

pipeline.train()
pre_res = pipeline.predict()
print(pre_res)
print('Done!')

# TODO: Collect "good" train/test data for cnn
# TODO: Integrate aws s3 storage for train/test data for models and for model loading for all models
# TODO: Move hard-coded cnn model compilation params to config.py
# TODO: Persist package versions in s3
# TODO: Remove unnecessary package import across repo