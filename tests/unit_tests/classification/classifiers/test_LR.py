import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from phishbench.classification.classifiers import LogisticRegression


class TestLR(unittest.TestCase):

    def test_fit(self):
        data = load_breast_cancer()
        lr = LogisticRegression("test")

        lr.fit(data.data, data.target)

        y_pred = lr.predict(data.data)
        self.assertGreater(accuracy_score(data.target, y_pred), 0.5)

    def test_fit_weighted(self):
        data = load_breast_cancer()
        lr = LogisticRegression("test")

        lr.fit_weighted(data.data, data.target)

        y_pred = lr.predict(data.data)
        self.assertGreater(accuracy_score(data.target, y_pred), 0.5)

    def test_param_search(self):
        data = load_breast_cancer()
        lr = LogisticRegression("test")

        result = lr.param_search(data.data, data.target)

        # Tests to make sure fitted classifier was stored
        y_pred = lr.predict(data.data)
        self.assertGreater(accuracy_score(data.target, y_pred), 0.5)

        # Tests to see
        self.assertIn("penalty", result)
