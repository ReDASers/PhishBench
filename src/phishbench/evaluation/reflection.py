from enum import Enum, unique
from functools import wraps


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
