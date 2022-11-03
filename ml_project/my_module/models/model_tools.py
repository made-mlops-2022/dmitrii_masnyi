from typing import Union, Dict

import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from my_module.entities import ModelParams

ClassifierModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(features: pd.DataFrame,
                target: pd.Series,
                train_params: ModelParams) -> ClassifierModel:
    if train_params.model_type == 'LogisticRegression':
        model = LogisticRegression()
    elif train_params.model_type == 'RandomForest':
        model = RandomForestClassifier()
    else:
        raise NotImplementedError('Use either LogReg or RF classifier')
    model.fit(features, target)
    return model


def predict_model(features: pd.DataFrame, model: ClassifierModel) -> np.ndarray:
    return model.predict(features)


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    metrics = {}
    metrics['accuracy_score'] = accuracy_score(target, predicts)
    metrics['f1_score'] = f1_score(target, predicts)
    metrics['recall_score'] = recall_score(target, predicts)
    metrics['precision_score'] = precision_score(target, predicts)
    return metrics


def save_model(model: ClassifierModel, path_to_save: str) -> str:
    with open(path_to_save, 'wb') as file:
        pickle.dump(model, file)
    return path_to_save


def load_model(path_to_load: str) -> ClassifierModel:
    with open(path_to_load, 'rb') as file:
        model = pickle.load(file)
    return model
