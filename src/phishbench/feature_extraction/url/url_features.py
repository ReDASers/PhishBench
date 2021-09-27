"""
This module handles the extraction of url features
"""
import time
from typing import List, Tuple, Dict, Optional

from tqdm import tqdm

from . import features as internal_features
from .. import settings
from ..reflection import load_features, FeatureClass, FeatureType
from ...input import url_input
from ...input.url_input import URLData
from ...utils import phishbench_globals


def create_new_features() -> List[FeatureClass]:
    """
    Gets URL features

    Returns
    -------
    features:
        A list of instantiated features
    """
    features = [x() for x in load_features(internal_features, 'URL')]
    print(f"Loaded {len(features)} features")
    return features


def extract_labeled_dataset(legit_dataset_folder: str, phish_dataset_folder: str,
                            features: Optional[List[FeatureClass]] = None):
    """
    Extract features from a labeled dataset split by files
    Parameters
    ----------
    legit_dataset_folder: str
        The path of the folder/file containing the legitimate urls
    phish_dataset_folder: str
        The path of the folder/file containing the phishing urls
    features: Optional[List[FeatureClass]]
        A list of feature objects or `None`. If `None`, then this function will load and instantiate new instances of
        the features

    Returns
    -------
    features: List[Dict]
        A list of dicts containing the extracted features
    labels: List[int]
        A list of labels. 0 is legitimate and 1 is phishing
    features: List[FeatureClass]
        The feature instances used.
    """
    download_url_flag = settings.feature_type_enabled(FeatureType.URL_NETWORK) or \
                        settings.feature_type_enabled(FeatureType.URL_WEBSITE)

    phishbench_globals.logger.info("Extracting email features. Legit: %s Phish: %s",
                                   legit_dataset_folder, phish_dataset_folder)

    print(f"Loading URLs from {legit_dataset_folder}")
    legit_urls, bad_url_list = url_input.read_dataset_url(legit_dataset_folder, download_url_flag)
    print(f"Loading URLs from {phish_dataset_folder}")
    phish_urls, bad_urls = url_input.read_dataset_url(phish_dataset_folder, download_url_flag)
    bad_url_list.extend(bad_urls)

    urls = legit_urls + phish_urls
    labels = [0] * len(legit_urls) + [1] * len(phish_urls)

    if features is None:
        features = create_new_features()
        for feature in features:
            feature.fit(urls, labels)

    print("Extracting features")
    feature_values = extract_features_list(urls, features)

    return feature_values, labels, features


def extract_features_list(urls: List[URLData], features: List[FeatureClass]):
    """
    Extracts features from a list of URLs

    Parameters
    ----------
    urls: List[URLData]
        The urls to extract features from
    features: List[FeatureClass]
        The features to extract
    Returns
    -------
    feature_list_dict: List[Dict[str]]
        A list of dicts containing the extracted features
    """

    feature_list_dict = []
    for url in tqdm(urls):
        feature_values, _ = extract_features_from_single(features, url)
        feature_list_dict.append(feature_values)

    return feature_list_dict


def extract_features_from_single(features: List[FeatureClass], url: URLData) -> Tuple[Dict, Dict]:
    """
    Extracts multiple features from a single url
    Parameters
    ----------
    features: List[FeatureClass]
        The features to extract
    url: URLData
        The url to extract the features from

    Returns
    -------
    feature_values: Dict
        The extracted feature values
    extraction_times: Dict
        The time it took to extract each feature
    """
    # pylint: disable=duplicate-code
    if not isinstance(url, URLData):
        raise TypeError("url must be an URLData object")

    dict_feature_values = {}
    dict_feature_times = {}

    for feature in features:
        result, ex_time = extract_single_feature_url(feature, url)
        if isinstance(result, dict):
            temp_dict = {feature.config_name + "." + key: value for key, value in result.items()}
            dict_feature_values.update(temp_dict)
        else:
            dict_feature_values[feature.config_name] = result
        dict_feature_times[feature.config_name] = ex_time

    return dict_feature_values, dict_feature_times


def extract_single_feature_url(feature: FeatureClass, url: URLData):
    """
    Extracts a single feature from a single url
    Parameters
    ----------
    feature
        The feature to extract
    url: URLData
        The url to extract the feature from

    Returns
    -------
    feature_value
        The value of the feature
    ex_time: float
        The time to extract the feature
    """
    # pylint: disable=broad-except
    # pylint: disable=duplicate-code
    if not isinstance(url, URLData):
        raise TypeError("url must be an URLData object")

    phishbench_globals.logger.debug(feature.config_name)
    start = time.process_time()
    try:
        feature_value = feature.extract(url)
    except Exception:
        phishbench_globals.logger.warning("Error extracting %s", feature.config_name, exc_info=True)
        feature_value = feature.default_value
    end = time.process_time()
    ex_time = end - start
    return feature_value, ex_time
