import inspect
from enum import Enum, unique
from functools import wraps
from typing import List, Callable

from scipy.sparse import issparse
import pandas as pd

from . import settings
from ..classification.core import BaseClassifier


@unique
class MetricType(Enum):
    PRED = 0
    PROB = 1
    CLUSTER = 2


def register_metric(metric_type: MetricType, config_name: str):
    def wrapped(function):
        @wraps(function)
        def wrapped_f(*args, **kwargs):
            return function(*args, **kwargs)

        wrapped_f.config_name = config_name
        wrapped_f.metric_type = metric_type
        return wrapped_f

    return wrapped


def load_metrics(source, filter_metrics=True):
    attrs = [getattr(source, x) for x in dir(source)]
    metrics = [x for x in attrs if inspect.isfunction(x) and hasattr(x, 'config_name')]
    if filter_metrics:
        metrics = [x for x in metrics if settings.is_enabled(x.config_name)]
    return metrics


def load_internal_metrics(filter_metrics=True) -> List[Callable]:
    from . import metrics
    return load_metrics(metrics, filter_metrics=filter_metrics)


def evaluate_classifier(classifier: BaseClassifier, x_test, y_test):
    """
    Evaluates a single classifier
    Parameters
    ----------
    classifier
        The classifier to evaluate.
    x_test
        The test features to evaluate with
    y_test
        The test labels to evaluate with

    Returns
    -------
        A dictionary containing the name of the metric and the corresponding scores
    """
    if issparse(x_test):
        x_test = x_test.toarray()
    metric_funcs: List[Callable] = load_internal_metrics()
    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)
    metrics = {}
    for metric in metric_funcs:
        if metric.metric_type == MetricType.PRED:
            metrics[metric.config_name] = metric(y_test, y_pred)
        else:
            metrics[metric.config_name] = metric(y_test, y_prob)
    return metrics


def evaluate_classifiers(classifiers: List[BaseClassifier], x_test, y_test):
    """
    Evaluates a set of classifiers
    Parameters
    ----------
    classifiers
        The classifiers to evaluate
    x_test
        The test features to evaluate with
    y_test
        The test labels to evaluate with

    Returns
    -------
    A pandas `DataFrame` containing the metrics of the classifiers
    """
    performance_list_dict = []
    for classifier in classifiers:
        metrics = evaluate_classifier(classifier, x_test, y_test)
        metrics['classifier'] = classifier.name
        performance_list_dict.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(performance_list_dict)
    columns: List = df.columns.tolist()
    columns.remove("classifier")
    columns.insert(0, "classifier")
    return df.reindex(columns=columns)
