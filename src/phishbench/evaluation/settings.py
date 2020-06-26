from ..utils import Globals

EVALUATION_SECTION = 'Evaluation Metrics'


def is_enabled(metric_name):
    return Globals.config[EVALUATION_SECTION].getboolean(metric_name)