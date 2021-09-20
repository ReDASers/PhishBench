"""
This module contains the core code for evaluating classifiers
"""
import inspect
import itertools
from typing import List, Dict

from scipy.sparse import issparse
import pandas as pd

from . import settings
from ..classification import BaseClassifier
from ..utils.reflection_utils import load_local_modules
from .reflection import MetricType, Metric
from . import metrics as internal_metrics


def load_metrics_from_module(source, filter_metrics: bool = True) -> List[Metric]:
    """
    Loads metrics from a module

    Parameters
    ----------
    source: ModuleType
        The module to import metrics from
    filter_metrics: bool
        Whether or not to filter out metrics according to the global config.

    Returns
    -------
        A list of metrics in the module.
    """
    attrs = [getattr(source, x) for x in dir(source)]
    metrics = [x for x in attrs if inspect.isfunction(x) and hasattr(x, 'metric_type')]
    if filter_metrics:
        metrics = [x for x in metrics if settings.is_enabled(x.config_name)]
    return metrics


def load_metrics(filter_metrics: bool = True) -> List[Metric]:
    """
    Loads all metrics

    Parameters
    ----------
    filter_metrics: bool
        Whether or not to filter the metrics

    Returns
    -------
        A list of feature functions
    """
    modules = load_local_modules()
    modules.append(internal_metrics)
    loaded_features = [load_metrics_from_module(module, filter_metrics) for module in modules]
    metrics = list(itertools.chain.from_iterable(loaded_features))
    return metrics


def evaluate_classifier(classifier: BaseClassifier, x_test, y_test) -> Dict[str, float]:
    """
    Evaluates a single classifier

    Parameters
    ----------
    classifier: A :py:class:`BaseClassifier <phishbench.classification.BaseClassifier>` object
        The classifier to evaluate.
    x_test: array-like of shape (n_samples, n_features)
        The test features to evaluate with
    y_test: array-like of shape (n_samples)
        The test labels to evaluate with

    Returns
    -------
    Dict[str, float]
        A dictionary mapping name of the metric to the corresponding score.
    """
    if issparse(x_test):
        x_test = x_test.toarray()
    metric_funcs: List[Metric] = load_metrics()
    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)
    metrics = {}
    for metric in metric_funcs:
        if metric.metric_type == MetricType.PRED:
            metrics[metric.config_name] = metric(y_test, y_pred)
        else:
            metrics[metric.config_name] = metric(y_test, y_prob)
    return metrics


def evaluate_classifiers(classifiers: List[BaseClassifier], x_test, y_test, verbose=1) -> pd.DataFrame:
    """
    Evaluates a set of classifiers

    Parameters
    ----------
    classifiers: A list of :py:class:`BaseClassifier <phishbench.classification.BaseClassifier>` objects
        The classifiers to evaluate
    x_test: array-like of shape (n_samples, n_features)
        The feature vectors of the test set.
    y_test: array-like of shape (n_samples)
        The labels of the test set with ``0`` being legitimate and ``1`` being phishing
    verbose: bool
        Whether or not to print progress to stdout.

            * ``0`` prints nothing.
            * ``1`` prints the classifiers being trained

    Returns
    -------
    A pandas :class:`DataFrame`
         The metrics of the classifiers.
    """
    performance_list_dict = []
    for classifier in classifiers:
        if verbose:
            print("Evaluating {}.".format(classifier.name))
        metrics = evaluate_classifier(classifier, x_test, y_test)
        metrics['classifier'] = classifier.name
        performance_list_dict.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(performance_list_dict)
    columns: List = df.columns.tolist()
    columns.remove("classifier")
    columns.insert(0, "classifier")
    return df.reindex(columns=columns)
