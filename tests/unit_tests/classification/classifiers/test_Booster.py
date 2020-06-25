import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from phishbench.classification.classifiers import Boosting


class TestBooster(unittest.TestCase):

    def test_fit(self):
        data = load_breast_cancer()
        booster = Boosting('test')
        booster.fit(data.data, data.target)
        y_pred = booster.predict(data.data)
        self.assertGreater(accuracy_score(data.target, y_pred), 0.5)

    def test_test_weighted(self):
        data = load_breast_cancer()
        booster = Boosting('test')
        booster.fit_weighted(data.data, data.target)
        y_pred = booster.predict(data.data)
        self.assertGreater(accuracy_score(data.target, y_pred), 0.5)

    def test_params_search(self):
        data = load_breast_cancer()
        booster = Boosting('test')
        result = booster.param_search(data.data, data.target)
        y_pred = booster.predict(data.data)
        self.assertGreater(accuracy_score(data.target, y_pred), 0.5)
        self.assertIn("n_estimators", result)
