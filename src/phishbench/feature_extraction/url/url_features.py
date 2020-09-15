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

    print("Loading URLs from {}".format(legit_path))
    legit_urls, bad_url_list = url_input.read_dataset_url(legit_path, download_url_flag)
    print("Loading URLs from {}".format(phish_path))
    phish_urls, bad_urls = url_input.read_dataset_url(phish_path, download_url_flag)
    bad_url_list.extend(bad_urls)

    features, corpus = extract_features_from_list_urls(legit_urls + phish_urls)
    labels = ([0] * len(legit_urls)) + ([1] * len(phish_urls))

    return features, labels, corpus


def extract_features_from_list_urls(urls: List[URLData]):
    """
    Extracts features from a list of URLs
    Parameters
    ----------
    urls: List[URLData]
        The urls to extract features from
    Returns
    -------
    feature_list_dict: List[Dict[str]]
        A list of dicts containing the extracted features
    corpus:
        A list of the downloaded websites
    """
    features = load_features(filter_features='URL', internal_features=internal_features)
    feature_list_dict = list()

    corpus = []
    for url in tqdm(urls):
        feature_values, _ = extract_features_from_single_url(features, url)
        feature_list_dict.append(feature_values)
        if settings.feature_type_enabled(FeatureType.URL_WEBSITE):
            soup = BeautifulSoup(url.downloaded_website, 'html5lib')
            corpus.append(str(soup))
    return feature_list_dict, corpus


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
