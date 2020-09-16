from enum import Enum, unique
from functools import wraps


@unique
class MetricType(Enum):
    PRED = 0
    PROB = 1
    CLUSTER = 2


class Metric:
    config_name: str
    metric_type: MetricType

    def __init__(self):
        raise SyntaxError('This is a stub for type hinting and should not be instantiated.')

    def __call__(self, y_test, y_pred) -> float:
        return 0


def register_metric(metric_type: MetricType, config_name: str):
    def wrapped(function):
        @wraps(function)
        def wrapped_f(*args, **kwargs):
            return function(*args, **kwargs)

        wrapped_f.config_name = config_name
        wrapped_f.metric_type = metric_type
        return wrapped_f

    return wrapped
