import numpy as np
import pandas as pd


class KNN:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y) -> None:
        self.X = X
        self.y = y

    def predict(self, X: pd.DataFrame) -> pd.Series:
        def _predict(x: pd.Series):
            return self.y.iloc[self.distance(x).nsmallest(self.k).index].mode()[0]
        return X.apply(_predict, axis=1)

    def distance(self, x: pd.Series) -> pd.Series | ValueError:
        if len(x) != len(self.X.columns):
            raise ValueError('Shape mismatch')
        return np.sqrt(np.sum((self.X - x) ** 2, axis=1)).rename('distance')


class Metrics:
    @staticmethod
    def accuracy(y: pd.Series, y_pred: pd.Series) -> float:
        return np.mean(y == y_pred)
