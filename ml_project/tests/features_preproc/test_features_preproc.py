import os
import unittest
import pandas as pd
import numpy as np
from my_module.features import split_data_and_target, create_transformer
from my_module.data import read_data
from my_module.entities import (
    FeatureParams,
    read_training_pipeline_params,
)


class TestFeaturesPreproc(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_to_data = "data/"
        self.path_to_csv = "data/heart_cleveland_upload.csv"
        self.path_to_training_config = "configs/train_configs/log_reg_train_config.yaml"

    def __assertIsFile(self):
        if not os.path.isfile(self.path_to_csv):
            raise AssertionError(f"File does not exist: {self.path_to_csv}")

    def __assertIsDirectory(self):
        if not os.path.exists(self.path_to_data):
            raise AssertionError(f"Directory does not exist: {self.path_to_data}")

    def test_data_exist(self):
        self.__assertIsDirectory()
        self.__assertIsFile()

    def test_split_data_target(self):
        data = read_data(self.path_to_csv)
        X, y = split_data_and_target(
            data,
            FeatureParams(
                target_col="condition",
                categorical_features="",
                numerical_features="",
                features_to_drop="",
                transformer_type="",
            ),
        )
        self.assertEqual(X.shape, (297, 13))
        self.assertEqual(y.shape, (297,))
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_standart_transformer(self):
        data = read_data(self.path_to_csv)
        training_dataclass = read_training_pipeline_params(self.path_to_training_config)
        X, _ = split_data_and_target(data, training_dataclass.feature_params)

        transformer = create_transformer(training_dataclass.feature_params)
        X_transf = transformer.fit_transform(X)
        self.assertEqual(X_transf.shape, (297, 28))

        num_feat_idx = []
        for idx in range(X_transf.shape[1]):
            if len(np.unique(X_transf[:, idx])):
                continue
            num_feat_idx.append(idx)

        for idx in range(X_transf.shape[1]):
            if idx not in num_feat_idx:
                continue
            self.assertAlmostEqual(X_transf[:, idx].mean(), 0, 2)
            self.assertAlmostEqual(X_transf[:, idx].std(), 0, 2)


if __name__ == "__main__":
    unittest.main()
