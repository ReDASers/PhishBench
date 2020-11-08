from ..utils import phishbench_globals

SECTION = 'Evaluation Metrics'


def is_enabled(metric_name):
    return phishbench_globals.config[SECTION].getboolean(metric_name)