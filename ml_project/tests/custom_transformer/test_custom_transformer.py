import os
import unittest
import numpy as np
from my_module.features import split_data_and_target, create_transformer
from my_module.data import read_data
from my_module.entities import read_training_pipeline_params


class TestCustomTransformer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_to_data = "data/"
        self.path_to_csv = "data/heart_cleveland_upload.csv"
        self.path_to_training_config_custom = (
            "configs/train_configs/log_reg_train_custom_transf_config.yaml"
        )
        self.path_to_training_config_default = (
            "configs/train_configs/log_reg_train_config.yaml"
        )

    def __assertIsFile(self):
        if not os.path.isfile(self.path_to_csv):
            raise AssertionError(f"File does not exist: {self.path_to_csv}")

    def __assertIsDirectory(self):
        if not os.path.exists(self.path_to_data):
            raise AssertionError(f"Directory does not exist: {self.path_to_data}")

    def test_data_exist(self):
        self.__assertIsDirectory()
        self.__assertIsFile()

    def test_custom_transformer(self):
        data = read_data(self.path_to_csv)
        training_dataclass_default = read_training_pipeline_params(
            self.path_to_training_config_default
        )
        training_dataclass_custom = read_training_pipeline_params(
            self.path_to_training_config_custom
        )

        X, _ = split_data_and_target(data, training_dataclass_default.feature_params)

        default_transformer = create_transformer(
            training_dataclass_default.feature_params
        )
        X_transf_default = default_transformer.fit_transform(X)

        custom_transformer = create_transformer(
            training_dataclass_custom.feature_params
        )
        X_transf_custom = custom_transformer.fit_transform(X)

        self.assertEqual(X_transf_custom.shape, X_transf_default.shape)
        self.assertEqual(X_transf_custom.shape, (297, 28))
        self.assertAlmostEquals(
            np.linalg.norm(X_transf_default - X_transf_custom) ** 2, 0, 2
        )


if __name__ == "__main__":
    unittest.main()
