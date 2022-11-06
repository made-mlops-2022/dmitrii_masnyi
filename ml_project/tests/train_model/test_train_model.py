import unittest
import os
import json
from my_module.features import split_data_and_target, create_transformer
from my_module.data import read_data
from my_module.entities import (
    FeatureParams,
    read_training_pipeline_params,
)
from my_module.train import train_pipeline


class TestModelTrain(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = read_training_pipeline_params(
            "configs/train_configs/log_reg_train_config.yaml"
        )

    def __assertIsModel(self):
        if not os.path.isfile(self.config.output_model_path):
            raise AssertionError(
                f"File does not exist: {self.config.output_model_path}"
            )

    def __assertIsMetrics(self):
        if not os.path.isfile(self.config.metric_path):
            raise AssertionError(f"File does not exist: {self.config.metric_path}")

    def test_model_saved(self):
        self.__assertIsModel()
        self.__assertIsMetrics()

    def test_quality(self):
        data = {}
        with open(self.config.metric_path) as json_file:
            data = json.load(json_file)
        self.assertGreater(data["accuracy_score"], 0.6)
        self.assertGreater(data["f1_score"], 0.6)
        self.assertGreater(data["recall_score"], 0.6)
        self.assertGreater(data["precision_score"], 0.6)


if __name__ == "__main__":
    params = read_training_pipeline_params(
        "configs/train_configs/log_reg_train_config.yaml"
    )
    train_pipeline(params)
    unittest.main()
