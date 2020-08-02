import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from phishbench.classification.classifiers import FeedForwardNN


class TestFFNN(unittest.TestCase):

    def test_predict(self):
        data = load_breast_cancer()
        clf = FeedForwardNN('test')
        clf.fit(data.data, data.target)

        # Here, we're primarily concerned that the output is correctly formatted and not wildly inaccurate
        y_pred = clf.predict(data.data)
        self.assertEqual(len(y_pred.shape), 1)
        self.assertTrue(all(map(lambda x: x in [0, 1], y_pred)))
        self.assertTrue(accuracy_score(data.target, y_pred) > 0.5)

    def test_predict_proba(self):
        data = load_breast_cancer()
        clf = FeedForwardNN('test')
        clf.fit(data.data, data.target)
        y_pred = clf.predict_proba(data.data)

        self.assertEqual(len(y_pred.shape), 1)
        self.assertTrue(all(map(lambda x: 1 >= x >= 0, y_pred)))

    # def test_param_search(self):
    #     data = load_breast_cancer()
    #     clf = FeedForwardNN('test')
    #     results = clf.param_search(data.data, data.target)
    #     self.assertTrue('activation' in results)
