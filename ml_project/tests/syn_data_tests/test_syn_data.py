from random import randint
import unittest
import numpy as np
import pandas as pd
from my_module.entities import read_predict_pipeline_params
from my_module.predict import predict_pipeline


class TestSyntData(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = "configs/predict_configs/lr_pred_config.yaml"
        self.columns = np.array(
            [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
            ]
        )

    def __generate_data(self):
        data = [
            randint(50, 70),
            randint(0, 1),
            randint(0, 3),
            randint(100, 180),
            randint(50, 350),
            randint(0, 1),
            randint(0, 2),
            randint(90, 170),
            randint(0, 1),
            randint(0, 9),
            randint(0, 2),
            randint(0, 3),
            randint(0, 2),
        ]
        return np.array(data)

    def test_synt_data_pred(self):
        data = pd.DataFrame([self.__generate_data()], columns=self.columns)
        data.to_csv("data/test_data/synt_data.csv", index=False)
        pred_params = read_predict_pipeline_params(self.config_path)
        pred_params.input_data_path = "data/test_data/synt_data.csv"
        pred_params.output_data_path = "predicts/synt_data.csv"
        predict_pipeline(pred_params)
        pred = pd.read_csv("predicts/synt_data.csv")
        self.assertEqual(pred.iloc[0, 0] == 0 or pred.iloc[0, 0] == 1, True)
        self.assertEqual(len(pred), 1)
        self.assertIsInstance(pred, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
