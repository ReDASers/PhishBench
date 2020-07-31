import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from phishbench.classification.classifiers import XGBoost


class TestXGB(unittest.TestCase):

    def test_fit(self):
        data = load_breast_cancer()
        xgb = XGBoost('test')
        xgb.fit(data.data, data.target)
        y_pred = xgb.predict(data.data)
        self.assertGreater(accuracy_score(data.target, y_pred), 0.5)

    def test_weighted(self):
        data = load_breast_cancer()
        xgb = XGBoost('test')
        xgb.fit_weighted(data.data, data.target)
        y_pred = xgb.predict(data.data)
        self.assertGreater(accuracy_score(data.target, y_pred), 0.5)

    def test_params_search(self):
        data = load_breast_cancer()
        xgb = XGBoost('test')
        result = xgb.param_search(data.data, data.target)
        y_pred = xgb.predict(data.data)
        self.assertGreater(accuracy_score(data.target, y_pred), 0.5)
        self.assertIn("booster", result)
