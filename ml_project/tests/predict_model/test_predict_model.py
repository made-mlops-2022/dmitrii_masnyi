import unittest
import os
import pandas as pd
from my_module.data import read_data
from my_module.entities import (
    read_training_pipeline_params,
    read_predict_pipeline_params,
)
from my_module.train import train_pipeline
from my_module.predict import predict_pipeline


class TestModelPredict(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_config = read_training_pipeline_params(
            "configs/train_configs/log_reg_train_config.yaml"
        )
        self.predict_config = read_predict_pipeline_params(
            "configs/predict_configs/lr_pred_config.yaml"
        )

    def __assertIsModel(self):
        if not os.path.isfile(self.train_config.output_model_path):
            raise AssertionError(
                f"File does not exist: {self.train_config.output_model_path}"
            )

    def __assertIsMetrics(self):
        if not os.path.isfile(self.train_config.metric_path):
            raise AssertionError(
                f"File does not exist: {self.train_config.metric_path}"
            )

    def __assertIsPreds(self):
        if not os.path.isfile(self.predict_config.output_data_path):
            raise AssertionError(
                f"File does not exist: {self.predict_config.output_data_path}"
            )

    def test_model_saved(self):
        self.__assertIsModel()
        self.__assertIsMetrics()

    def test_predicts_saved(self):
        self.__assertIsPreds()

    def test_preds_shape(self):
        test_data = read_data(self.predict_config.input_data_path)
        preds = read_data(self.predict_config.output_data_path)
        self.assertEqual(test_data.shape[0], preds.shape[0])

    def test_pred_type(self):
        preds = read_data(self.predict_config.output_data_path)
        self.assertIsInstance(preds, pd.DataFrame)


if __name__ == "__main__":
    params = read_training_pipeline_params(
        "configs/train_configs/log_reg_train_config.yaml"
    )
    train_pipeline(params)
    params = read_predict_pipeline_params("configs/predict_configs/lr_pred_config.yaml")
    predict_pipeline(params)
    unittest.main()
