import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.utils.estimator_checks import check_estimator

from phishbench.classification.classifiers import ExtremeLearningMachine
from phishbench.classification.classifiers.elm import ELMClassifier


class TestELM(unittest.TestCase):

    def test_predict(self):
        data = load_breast_cancer()
        elm = ExtremeLearningMachine('test')
        elm.fit(data.data, data.target)

        # Here, we're primarily concerned that the output is correctly formatted and not wildly inaccurate
        y_pred = elm.predict(data.data)
        self.assertTrue(all(map(lambda x: x in [0, 1], y_pred)))
        self.assertTrue(accuracy_score(data.target, y_pred) > 0.5)

    def test_predict_proba(self):
        data = load_breast_cancer()
        elm = ExtremeLearningMachine('test')
        elm.fit(data.data, data.target)
        y_pred = elm.predict_proba(data.data)
        self.assertTrue(all(map(lambda x: 1 >= x >= 0, y_pred)))

    def test_param_search(self):
        data = load_breast_cancer()
        elm = ExtremeLearningMachine('test')
        results = elm.param_search(data.data, data.target)
        self.assertTrue('activation' in results)
