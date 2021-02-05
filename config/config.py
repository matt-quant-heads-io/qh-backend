

CONFIG_PARAMS = {
    "api_creds": {
        "polygon": {
            "api_key": "RzpEOwfKImP4AQOyJhCBPfwpIl4N_iY5"
        }
    },
    "runtime_variables": {
        "verbose": False
    },
    "parent_key": "models",
    "models": {
        # THIS IS COMPLETE PIPELINE
        # "pipeline": [("library.models.recurrent_nn", "RecurrentNNWrapper"),
        #                      ("library.models.convolutional_nn", "ConvolutionalNNWrapper"),
        #                      ("logistic_regression", "LogisticRegressionWrapper"),
        #                      ("library.models.bayes_nn", "BayesNNWrapper")],
        "pipeline": [("library.models.recurrent_nn", "RecurrentNNWrapper"),
                     ("library.models.convolutional_nn", "ConvolutionalNNWrapper")],
        # TODO: configure "model_instances_dir" to work with AWS bucket
        "convolutional_nn": {
            "model_config": {
                "lookback_period": 25,
                "periodicity": 1,
                "optimizer": "rmsprop",
                "loss": "categorical_crossentropy",
                "patterns": [
                    "Channel Down",
                    "Channel Up",
                    "Double Bottom",
                    "Double Top",
                    "Head & Shoulders",
                    "Inv. Head & Shoulders",
                    "Triangle Ascending",
                    "Triangle Descending",
                    "Wedge Up",
                    "Wedge Down",
                ],
            },
            # TODO: configure s3 with chart images
            "symbol": "AAPL",
            "predict_data_path": "/Users/mattscomputer/qh_data_access",
            "load_model_from_file": True,
            "train_data_path": "/Users/mattscomputer/qh_data_access/cnn_data",
            "test_data_path": "/Users/mattscomputer/qh_data_access/test_data",
            "pred_file": "pred_image.jpg",
            "persisted_model_path": "/Users/mattscomputer/qh_data_access/cnn_model.h5"
        },
        "recurrent_nn": {
            # TODO: configure s3 with csv data
            "model_config": {
                "output_units": 1,
                "lr": 0.001,
                "loss": 'mse',
                "num_units": 1000,
                "activation_func": 'relu',
                "batch_size": 32,
                "epochs": 5,
                "dropout": 0.1,
                "alpha": 0.5,
                "optimizer": 'adam',
                "num_inputs": 4, #3,  # this is the looback_period
                "num_features": 23,
                "train_split_size": 0.99,
                "scale_data": True
            },
            "predict_data_path": "/Users/mattscomputer/qh_data_access",
            "load_model_from_file": True,
            "train_data_path": ".",
            "train_file": "train_data_df.csv",
            "predict_file": "predict_data_df.csv",
            "persisted_model_path": "/Users/mattscomputer/qh_data_access/model_tf_120.h5"
        },
        # TODO: Remove below what is not needed:
        "ec2_host": "",
        "models_s3_bucket": ""
    }
}


class Config(object):
    def __init__(self):
        self.params = CONFIG_PARAMS

    def __getitem__(self, item):
        try:
            return self.params.get(item, None)
        except KeyError as key_error:
            raise key_error
