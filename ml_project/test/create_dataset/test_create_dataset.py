import os
import unittest
import pandas as pd
from my_module.data import read_data, split_train_val_data
from my_module.entities import SplittingParams


class TestDatasetCreation(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_to_data = 'data/'
        self.path_to_csv = 'data/heart_cleveland_upload.csv'

    def __assertIsFile(self):
        if os.path.isfile(self.path_to_data):
            raise AssertionError(f'File does not exist: {str(self.path_to_csv)}')

    def __assertIsDirectory(self):
        if os.path.exists(self.path_to_data):
            raise AssertionError(f'Directory does not exist: {str(self.path_to_data)}')

    def test_data_exist(self):
        self.__assertIsDirectory()
        self.__assertIsFile()

    def test_data_load(self):
        data = read_data(f'../../{self.path_to_csv}')
        self.assertIsInstance(data, pd.DataFrame)

    def test_split(self):
        data = read_data(f'../../{self.path_to_csv}')
        train, val = split_train_val_data(data, SplittingParams(val_size = 0.3,random_state = 42))
        self.assertEqual(train.shape, (207, 14))
        self.assertEqual(val.shape, (90, 14))


if __name__ == '__main__':
    unittest.main()
