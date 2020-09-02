import time
from typing import List, Callable, Tuple, Dict

from bs4 import BeautifulSoup
from tqdm import tqdm

from . import features as internal_features
from .. import settings
from ..reflection import load_features, FeatureType
from ... import Features
from ... import dataset
from ...Features_Support import Cleaning, read_alexa
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

    alexa_data = {}
    if phishbench_globals.config['URL_Feature_Types'].getboolean("HTML") and \
            phishbench_globals.config["HTML_Features"].getboolean("ranked_matrix"):
        alexa_path = phishbench_globals.config["Support Files"]["path_alexa_data"]
        alexa_data = read_alexa(alexa_path)

    corpus = []
    for url in tqdm(urls):
        try:
            feature_values, _ = url_features(url, corpus, alexa_data, features)
            feature_list_dict.append(feature_values)
        except Exception:
            error_string = "Error extracting features from {}".format(url.raw_url)
            phishbench_globals.logger.warning(error_string, exc_info=True)
            bad_url_list.append(url.raw_url)
    return feature_list_dict, corpus


def url_features(url: URLData, corpus, alexa_data, features):
    dict_feature_values, dict_extraction_times = extract_features_from_single_url(features, url)
    phishbench_globals.logger.debug("rawurl: %s", str(url))

    if settings.feature_type_enabled(FeatureType.URL_RAW):
        single_url_feature(url.raw_url, dict_feature_values, dict_extraction_times)
        phishbench_globals.logger.debug("url_features >>>>>> complete")

    if settings.feature_type_enabled(FeatureType.URL_NETWORK):
        single_network_features(url, dict_feature_values, dict_extraction_times)
        phishbench_globals.logger.debug("network_features >>>>>> complete")

    if settings.feature_type_enabled(FeatureType.URL_WEBSITE):
        single_url_html_features(url, alexa_data, dict_feature_values, dict_extraction_times)
        phishbench_globals.logger.debug("html_features >>>>>> complete")
        soup = BeautifulSoup(url.downloaded_website, 'html5lib')

        if settings.feature_type_enabled(FeatureType.URL_WEBSITE_JAVASCRIPT):
            single_javascript_features(soup, url.downloaded_website, dict_feature_values, dict_extraction_times)
            phishbench_globals.logger.debug("javascript features >>>>>> complete")

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


def single_url_feature(raw_url, list_features, list_time):
    phishbench_globals.logger.debug("Extracting single url features from %s", raw_url)

    Features.URL_special_char_count(raw_url, list_features, list_time)

    Features.URL_Has_More_than_3_dots(raw_url, list_features, list_time)

    Features.URL_Has_anchor_tag(raw_url, list_features, list_time)

    Features.URL_Token_Count(raw_url, list_features, list_time)

    Features.URL_Average_Path_Token_Length(raw_url, list_features, list_time)

    Features.URL_Average_Domain_Token_Length(raw_url, list_features, list_time)

    Features.URL_Longest_Domain_Token(raw_url, list_features, list_time)

    Features.URL_Protocol_Port_Match(raw_url, list_features, list_time)

    Features.URL_Has_WWW_in_Middle(raw_url, list_features, list_time)

    Features.URL_Has_Hex_Characters(raw_url, list_features, list_time)

    Features.URL_Double_Slashes_Not_Beginning_Count(raw_url, list_features, list_time)

    Features.URL_Brand_In_Url(raw_url, list_features, list_time)

    Features.URL_Is_Whitelisted(raw_url, list_features, list_time)


def single_url_html_features(url: URLData, alexa_data, list_features, list_time):
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')

    phishbench_globals.logger.debug("Extracting single html features from %s", url.raw_url)

    Features.HTML_ranked_matrix(soup, url.raw_url, alexa_data, list_features, list_time)

    Features.HTML_LTree_Features(soup, url.raw_url, list_features, list_time)

    Features.HTML_number_suspicious_content(soup, list_features, list_time)

    Features.HTML_inbound_count(soup, url.raw_url, list_features, list_time)

    Features.HTML_outbound_count(soup, url.raw_url, list_features, list_time)

    Features.HTML_inbound_href_count(soup, url.raw_url, list_features, list_time)

    Features.HTML_outbound_href_count(soup, url.raw_url, list_features, list_time)

    # TODO: Reimplement as reflection features
    #
    # Features.HTML_Is_Login(downloaded_website.html, raw_url, list_features, list_time)


def single_javascript_features(soup, html, list_features, list_time):
    phishbench_globals.logger.debug("Extracting single javascript features")

    Features.Javascript_number_of_exec(soup, list_features, list_time)

    Features.Javascript_number_of_escape(soup, list_features, list_time)

    Features.Javascript_number_of_eval(soup, list_features, list_time)

    Features.Javascript_number_of_link(soup, list_features, list_time)

    Features.Javascript_number_of_unescape(soup, list_features, list_time)

    Features.Javascript_number_of_search(soup, list_features, list_time)

    Features.Javascript_number_of_setTimeout(soup, list_features, list_time)

    Features.Javascript_number_of_iframes_in_script(soup, list_features, list_time)

    Features.Javascript_number_of_event_attachment(soup, list_features, list_time)

    Features.Javascript_rightclick_disabled(html, list_features, list_time)

    Features.Javascript_number_of_total_suspicious_features(list_features, list_time)


def single_network_features(url, list_features, list_time):
    phishbench_globals.logger.debug("Extracting network features from %s", url)
    Features.Network_creation_date(url.domain_whois, list_features, list_time)

    Features.Network_expiration_date(url.domain_whois, list_features, list_time)

    Features.Network_updated_date(url.domain_whois, list_features, list_time)

    Features.Network_as_number(url.ip_whois, list_features, list_time)

    Features.Network_number_name_server(url.dns_results, list_features, list_time)

    Features.Network_dns_ttl(url.raw_url, list_features, list_time)

    Features.Network_DNS_Info_Exists(url.raw_url, list_features, list_time)
