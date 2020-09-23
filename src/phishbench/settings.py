from .utils import phishbench_globals

PB_SECTION = 'PhishBench Sections'


DEFAULT_SETTINGS = {
    'Mode': 'URL ; Options are "URL" or "Email"'
}


def mode():
    mode_str = phishbench_globals.config[PB_SECTION].strip().get('Mode')
    if mode_str.lower().startswith('url'):
        return 'URL'
    if mode_str.lower().startswith('email'):
        return 'Email'
    raise ValueError('Mode must either be email or url')
