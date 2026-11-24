"""Platt scaling on raw bonafide scores s = logit[1] - logit[0]. Fit ONLY on non-locked dev data."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = a, b

    def fit(self, scores, labels):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(scores, float).reshape(-1, 1), np.asarray(labels))
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def predict_proba(self, s):
        return 1.0 / (1.0 + np.exp(-(self.a * np.asarray(s, float) + self.b)))

    def save(self, path):
        json.dump({"a": self.a, "b": self.b}, open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls(**json.load(open(path)))
