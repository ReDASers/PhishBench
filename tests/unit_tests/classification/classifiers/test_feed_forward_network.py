"""
The test for `phishbench.classification.classifiers.FeedForwardNN`
"""
import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from phishbench.classification.classifiers import FeedForwardNN


class TestFFNN(unittest.TestCase):

    def test_predict(self):
        """
        Tests the `FeedForwardNN.predict` to make sure that

        1. It returns a 1 dimensional vector
        2. All predictions are either 0 or 1
        3. It's better than random chance.

        """
        x, y = load_breast_cancer(return_X_y=True)
        clf = FeedForwardNN('test')
        clf.fit(x, y)

        # Here, we're primarily concerned that the output is correctly formatted and not wildly inaccurate
        y_pred = clf.predict(x)
        self.assertEqual(len(y_pred.shape), 1)
        self.assertTrue(all(map(lambda y_hat: y_hat in [0, 1], y_pred)))
        self.assertTrue(accuracy_score(y, y_pred) > 0.5)

    def test_predict_proba(self):
        """
        Tests the `FeedForwardNN.predict_proba` function to make sure that

        1. It returns at 1d vector
        2. All probabilities are between 0 and 1

        """
        x, y = load_breast_cancer(return_X_y=True)
        clf = FeedForwardNN('test')
        clf.fit(x, y)
        y_pred = clf.predict_proba(x)

        self.assertEqual(len(y_pred.shape), 1)
        self.assertTrue(all(map(lambda y_hat: 1 >= y_hat >= 0, y_pred)))

    # def test_param_search(self):
    #     data = load_breast_cancer()
    #     clf = FeedForwardNN('test')
    #     results = clf.param_search(data.data, data.target)
    #     self.assertTrue('activation' in results)
