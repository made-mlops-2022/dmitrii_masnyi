from sklearn.base import BaseEstimator, TransformerMixin


class CustomStandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std()
        return self

    def transform(self, data):
        data_copy = data.copy()
        data_copy = (data_copy - self.mean) / self.std
        return data_copy
