from ..utils import globals

EVALUATION_SECTION = 'Evaluation Metrics'


def is_enabled(metric_name):
    return globals.config[EVALUATION_SECTION].getboolean(metric_name)