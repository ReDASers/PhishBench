import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from phishbench.classification.classifiers import Boosting


class TestBoosting(unittest.TestCase):

    def test_fit(self):
        data = load_breast_cancer()
        boosting = Boosting('test')
        boosting.fit(data.data, data.target)
        y_pred = boosting.predict(data.data)
        self.assertGreater(accuracy_score(data.target, y_pred), 0.5)

    def test_test_weighted(self):
        data = load_breast_cancer()
        boosting = Boosting('test')
        boosting.fit_weighted(data.data, data.target)
        y_pred = boosting.predict(data.data)
        self.assertGreater(accuracy_score(data.target, y_pred), 0.5)

    def test_params_search(self):
        data = load_breast_cancer()
        boosting = Boosting('test')
        result = boosting.param_search(data.data, data.target)
        y_pred = boosting.predict(data.data)
        self.assertGreater(accuracy_score(data.target, y_pred), 0.5)
        self.assertIn("n_estimators", result)
