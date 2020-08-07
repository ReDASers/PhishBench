from configparser import ConfigParser
from bs4 import BeautifulSoup
import pathlib
import os


def get_mock_config() -> ConfigParser:
    config = ConfigParser()
    config['Features'] = {}
    config['HTML_Features'] = {}
    config['URL_Features'] = {}
    config['URL_Features']['url_token_delimiter'] = '.'
    config['Network_Features'] = {}
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


def get_soup(filename) -> BeautifulSoup:
    current_file_folder = pathlib.Path(__file__).parent.absolute()
    test_file = os.path.join(current_file_folder, 'mock_webpages', filename)
    with open(test_file) as f:
        soup = BeautifulSoup(f.read(), features="lxml")
    return soup
