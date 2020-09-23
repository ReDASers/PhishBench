from .utils import phishbench_globals

PB_SECTION = 'PhishBench Sections'


DEFAULT_SETTINGS = {
    'Mode': 'URL ; Options are "URL" or "Email"'
}


def mode():
    mode_str = phishbench_globals.config[PB_SECTION].get('Mode').strip()
    if mode_str.lower().startswith('url'):
        return 'URL'
    if mode_str.lower().startswith('email'):
        return 'Email'
    raise ValueError('Mode must either be email or url')


def feature_extraction() -> bool:
    """
    Whether or not to extract features
    """
    return phishbench_globals.config[PB_SECTION].getboolean('feature extraction')