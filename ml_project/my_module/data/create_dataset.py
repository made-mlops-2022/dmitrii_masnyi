import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from my_module.entities import SplittingParams


def read_data(path_to_data: str) -> pd.DataFrame:
    data = pd.read_csv(path_to_data)
    return data


def split_train_val_data(
    data: pd.DataFrame, split_params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        data, test_size=split_params.val_size, random_state=split_params.random_state
    )
    return train_df, val_df
