from ..utils import phishbench_globals

SECTION_NAME = "Preprocessing"
DEFAULTS = {
    'min_max_scaling': 'True',
    'dataset balancing': 'True',
    'feature selection': 'True'
}


def min_max_scaling() -> bool:
    """
    Whether or not to perform min max scaling
    """
    return phishbench_globals.config[SECTION_NAME].getboolean('min_max_scaling')


def dataset_balancing() -> bool:
    """
    Whether or not to balance the dataset
    """
    return phishbench_globals.config[SECTION_NAME].getboolean('dataset balancing')


def feature_selection() -> bool:
    """
    Whether or not to perform feature selection
    """
    return phishbench_globals.config[SECTION_NAME].getboolean('feature selection')
