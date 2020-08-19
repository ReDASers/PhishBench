"""
This module contains the core code for evaluating classifiers
"""
import inspect
import itertools
from typing import List, Callable

from scipy.sparse import issparse
import pandas as pd

from . import settings
from ..classification.core import BaseClassifier
from ..utils.reflection_utils import load_local_modules
from .reflection import MetricType
from . import metrics as internal_metrics


def load_metrics_from_module(source, filter_metrics=True):
    """
    Loads metrics from a module
    Parameters
    ----------
    source: ModuleType
        The module to import metrics from
    filter_metrics: Union[str, None]
        Whether or not to load metrics based on `phishbench.utils.phishbench_globals.config`

    Returns
    -------
    A list of metrics in the module.
    """
    attrs = [getattr(source, x) for x in dir(source)]
    metrics = [x for x in attrs if inspect.isfunction(x) and hasattr(x, 'metric_type')]
    if filter_metrics:
        metrics = [x for x in metrics if settings.is_enabled(x.config_name)]
    return metrics


def load_metrics(filter_metrics=True) -> List[Callable]:
    """
    Loads all metrics

    Parameters
    ----------
    filter_metrics: Union[str, None]
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
    metric_funcs: List[Callable] = load_metrics()
    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)
    metrics = {}
    for metric in metric_funcs:
        if metric.metric_type == MetricType.PRED:
            metrics[metric.config_name] = metric(y_test, y_pred)
        else:
            metrics[metric.config_name] = metric(y_test, y_prob)
    return metrics


def evaluate_classifiers(classifiers: List[BaseClassifier], x_test, y_test, verbose=1):
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
    verbose: bool
        Whether or not to print progress to stdout.
        `0` prints nothing. `1` prints the classifiers being trained
    Returns
    -------
    A pandas `DataFrame` containing the metrics of the classifiers
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
