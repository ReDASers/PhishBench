from configparser import ConfigParser


def get_mock_config() -> ConfigParser:
    config = ConfigParser()
    config['Features'] = {}
    config['HTML_Features'] = {}
    config['URL_Features'] = {}
    config['URL_Features']['url_token_delimiter'] = '.'
    config['Network_Features']={}
    config['Javascript_Features'] = {}
    config['Classifiers'] = {}
    config['Imbalanced Datasets'] = {}
    config['Evaluation Metrics'] = {}
    config['Preprocessing'] = {}
    config["Feature Selection"] = {}
    config['Dataset Path'] = {}
    config['Email or URL feature Extraction'] = {}
    config['Extraction'] = {}
    config['Features Format'] = {}
    config['Classification'] = {}
    config["Summary"] = {}
    return config
