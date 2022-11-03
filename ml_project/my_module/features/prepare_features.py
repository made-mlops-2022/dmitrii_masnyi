from typing import Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from my_module.entities import FeatureParams
from my_module.transformer import CustomStandardScaler


def split_data_and_target(
        data_frame: pd.DataFrame, feature_params: FeatureParams
) -> Tuple[pd.DataFrame, pd.Series]:
    target = data_frame[feature_params.target_col]
    data = data_frame.drop(feature_params.target_col, axis=1)
    return data, target

def create_transformer(feature_params: FeatureParams) -> Pipeline:
    if feature_params.transformer_type == 'custom':
        transformer = ColumnTransformer([
            ('numerical_preprocessing', CustomStandardScaler(), feature_params.numerical_features),
            ('categorical_preprocessing', OneHotEncoder(), feature_params.categorical_features)
            ])
    elif feature_params.transformer_type == 'default':
        transformer = ColumnTransformer([
            ('numerical_preprocessing', StandardScaler(), feature_params.numerical_features),
            ('categorical_preprocessing', OneHotEncoder(), feature_params.categorical_features)
            ])
    else:
        transformer = ColumnTransformer([
            ('categorical_preprocessing', OneHotEncoder(), feature_params.categorical_features)
            ])
    return transformer
