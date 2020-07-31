from ..utils import phishbench_globals

EVALUATION_SECTION = 'Evaluation Metrics'


def is_enabled(metric_name):
    return phishbench_globals.config[EVALUATION_SECTION].getboolean(metric_name)