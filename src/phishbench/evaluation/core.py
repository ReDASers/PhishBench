import inspect
from enum import Enum, unique
from functools import wraps
from typing import List, Callable

from ..classification.core import BaseClassifier


@unique
class MetricType(Enum):
    PRED = 0
    PROB = 1
    CLUSTER = 2


def register_metric(type: MetricType, config_name: str):
    def wrapped(fn):
        @wraps(fn)
        def wrapped_f(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapped_f.config_name = config_name
        wrapped_f.metric_type = type
        return wrapped_f

    return wrapped


def load_metrics(source, filter=True):
    attrs = [getattr(source, x) for x in dir(source)]
    metrics = [x for x in attrs if inspect.isfunction(x) and hasattr(x, 'config_name')]
    if filter:
        #: TODO Implement filtering
        pass
    return metrics


def load_internal_metrics() -> List[Callable]:
    from . import metrics
    return load_metrics(metrics)


def evaluate_classifier(classifier: BaseClassifier, x_test, y_test):
    metric_funcs: List[Callable] = load_internal_metrics()
    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)[:, 1]
    print(y_prob.shape)
    metrics = {}
    for metric in metric_funcs:
        if metric.metric_type == MetricType.PRED:
            metrics[metric.config_name] = metric(y_test, y_pred)
        else:
            metrics[metric.config_name] = metric(y_test, y_prob)

    return metrics
