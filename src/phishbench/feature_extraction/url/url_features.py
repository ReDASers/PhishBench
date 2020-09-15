"""
This module handles the extraction of url features
"""
import time
from typing import List, Callable, Tuple, Dict

from bs4 import BeautifulSoup
from tqdm import tqdm

from . import features as internal_features
from .. import settings
from ..reflection import load_features, FeatureType
from ... import Features
from ... import dataset
from ...Features_Support import Cleaning
from ...input import url_input
from ...input.url_input import URLData
from ...utils import phishbench_globals


def Extract_Features_Urls_Testing():
    features, labels, corpus = extract_labeled_dataset(dataset.test_legit_path(), dataset.test_phish_path())
    print("Cleaning features")
    Cleaning(features)
    return features, labels, corpus


def Extract_Features_Urls_Training():
    features, labels, corpus = extract_labeled_dataset(dataset.train_legit_path(), dataset.train_phish_path())
    print("Cleaning features")
    Cleaning(features)
    return features, labels, corpus


def extract_labeled_dataset(legit_path, phish_path):
    download_url_flag = settings.feature_type_enabled(FeatureType.URL_NETWORK) or \
                        settings.feature_type_enabled(FeatureType.URL_WEBSITE)
    bad_url_list = []

    print("Extracting Features from {}".format(legit_path))
    legit_urls, bad_urls = url_input.read_dataset_url(legit_path, download_url_flag)
    bad_url_list.extend(bad_urls)
    legit_features, legit_corpus = extract_url_features(legit_urls, bad_url_list)

    print("Extracting Features from {}".format(phish_path))
    phish_urls, bad_urls = url_input.read_dataset_url(phish_path, download_url_flag)
    bad_url_list.extend(bad_urls)
    phish_features, phish_corpus = extract_url_features(phish_urls, bad_url_list)

    features = legit_features + phish_features
    labels = ([0] * len(legit_features)) + ([1] * len(phish_features))
    corpus = legit_corpus + phish_corpus

    return features, labels, corpus


def extract_url_features(urls: List[URLData], bad_url_list):
    features = load_features(filter_features='URL', internal_features=internal_features)
    feature_list_dict = list()

    corpus = []
    for url in tqdm(urls):
        try:
            feature_values, _ = url_features(url, corpus, features)
            feature_list_dict.append(feature_values)
        except Exception:
            error_string = "Error extracting features from {}".format(url.raw_url)
            phishbench_globals.logger.warning(error_string, exc_info=True)
            bad_url_list.append(url.raw_url)
    return feature_list_dict, corpus


def url_features(url: URLData, corpus, features):
    dict_feature_values, dict_extraction_times = extract_features_from_single_url(features, url)
    phishbench_globals.logger.debug("rawurl: %s", str(url))

    if settings.feature_type_enabled(FeatureType.URL_WEBSITE):
        soup = BeautifulSoup(url.downloaded_website, 'html5lib')
        corpus.append(str(soup))

    return dict_feature_values, dict_extraction_times


def extract_features_from_single_url(features: List[Callable], url: URLData) -> Tuple[Dict, Dict]:
    """
    Extracts multiple features from a single email
    Parameters
    ----------
    features: List
        The features to extract
    url: URLData
        The email to extract the features from

    Returns
    -------
    feature_values: Dict
        The extracted feature values
    extraction_times: Dict
        The time it took to extract each feature
    """
    dict_feature_values = dict()
    dict_feature_times = dict()

    for feature in features:
        result, ex_time = extract_single_feature_url(feature, url)
        if isinstance(result, dict):
            temp_dict = {feature.config_name + "." + key: value for key, value in result.items()}
            dict_feature_values.update(temp_dict)
        else:
            dict_feature_values[feature.config_name] = result
        dict_feature_times[feature.config_name] = ex_time

    return dict_feature_values, dict_feature_times


def extract_single_feature_url(feature: Callable, url: URLData):
    """
    Extracts a single feature from a single email
    Parameters
    ----------
    feature
        The feature to extract
    url: URLData
        The email to extract the feature from

    Returns
    -------
    feature_value
        The value of the feature
    ex_time: float
        The time to extract the feature
    """
    phishbench_globals.logger.debug(feature.config_name)
    start = time.process_time()
    try:
        feature_value = feature(url)
    except Exception:
        error_string = "Error extracting {}".format(feature.config_name)
        phishbench_globals.logger.warning(error_string, exc_info=True)
        feature_value = -1
    end = time.process_time()
    ex_time = end - start
    return feature_value, ex_time
