from ..classification.core import BaseClassifier
from sklearn.metrics import accuracy_score
from .core import register_metric, MetricType


@register_metric(MetricType.PRED, 'accuracy')
def accuracy(clf: BaseClassifier, x_test, y_test):
    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred)
