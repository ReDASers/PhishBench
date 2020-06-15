from enum import Enum, unique
from functools import wraps

from ..classification.core import BaseClassifier
from typing import List, Callable
import inspect


@unique
class MetricType(Enum):
    PRED = 0
    PROB = 1


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
    metrics: List[Callable] = load_internal_metrics()
    print(metrics)
    results = {metric.__name__: metric(classifier, x_test, y_test) for metric in metrics}
    return results
