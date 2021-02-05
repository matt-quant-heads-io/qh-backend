import sys
import pandas as pd
from keras.preprocessing import image
import numpy as np

from config.config import Config
from utils.utils import load_images, convert_class_path_to_file_name

CONFIG = Config()
MODELS_DETAILS = CONFIG[CONFIG["parent_key"]]


class Pipeline(object):
    def __init__(self, verbose=False):
        self._verbose = verbose
        self.models_details = MODELS_DETAILS
        self._config = None
        self._models_pipeline = self._instantiate_pipeline_models()
        # self._s3_resource = s3_resource

    def _instantiate_pipeline_models(self):
        try:
            pipeline = self.models_details["pipeline"]
            if len(pipeline) == 0 or pipeline is None:
                msg = "No model objects defined in 'pipeline' Config key"
                raise Exception(msg)
            models_pipeline = []
            for import_path, model in pipeline:
                file_name = convert_class_path_to_file_name(import_path, verbose=self._verbose)
                model_build_instructions = self.models_details[file_name]
                model = getattr(sys.modules[import_path], model)(model_details=model_build_instructions)
                if model_build_instructions["load_model_from_file"]:
                    model.load_model(verbose=self._verbose)
                else:
                    model.build_model(verbose=self._verbose)
                models_pipeline.append(model)
            return models_pipeline
        except Exception as e:
            msg = "Error in __build_models_pipeline"
            raise e(msg)

    def _load_pred_data(self, pred_data_path=None):
        try:
            if pred_data_path is None:
                msg = "pred_data_path cannot be None type"
                raise Exception(msg)
            model_input_values = None
            if 'csv' in pred_data_path:
                # TODO: make sure this is grabbing only the most recent row of the df csv stream (sorted by timestamp etc.)
                model_input_values = pd.read_csv(pred_data_path)[0].values
            elif 'jpg' in pred_data_path:
                # TODO: make sure this is grabbing only the most recent row of the df csv stream (sorted by timestamp etc.)
                path = load_images(self._s3_resource, pred_data_path) #TODO: make sure load_images downloads the file locally and returns the file path of the local file
                img = image.load_img(path, target_size=(300, 300))
                x = image.img_to_array(img)
                model_input_values = np.expand_dims(x, axis=0)
                #TODO: add logic to delete the locally downloaded input file now that we have the prediction
            return {'input_data': model_input_values}
        except Exception as e:
            msg = "Error in _load_pred_data"
            raise e(msg)

    # TODO: refactor this so we can use non default function params
    def train(self):
        try:
            pipeline = self.models_details["pipeline"]
            if len(pipeline) == 0 or pipeline is None:
                msg = "No model objects defined in 'pipeline' Config key"
                raise Exception(msg)
            for model in self._models_pipeline:
                training_data_tuple = model.get_training_data()
                model.train(training_data_tuple)
        except Exception as e:
            msg = "Error in train"
            raise e(msg)

    def predict(self):
        try:
            if len(self._models_pipeline) == 0 or self._models_pipeline is None:
                msg = "self._models_pipeline must be a non-empty list"
                raise Exception(msg)
            predictions = []
            for model in self._models_pipeline:
                pred_data = model.get_pred_data()
                pred_result = model.predict(pred_data=pred_data)
                predictions.append({f"{model._model_name_}": pred_result})
            return predictions
        except Exception as e:
            msg = "Error in train()"
            raise e(msg)



