import traceback
from typing import List

from bs4 import BeautifulSoup
from tqdm import tqdm

from phishbench import Features
from phishbench.Features_Support import Cleaning, read_alexa
from phishbench.input import input as pb_input
from phishbench.input.url_input import URLData
from phishbench.utils import phishbench_globals
from ... import dataset


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
    download_url_flag = phishbench_globals.config['URL_Feature_Types'].getboolean('Network') or \
                        phishbench_globals.config['URL_Feature_Types'].getboolean('HTML')
    bad_url_list = []

    print("Extracting Features from {}".format(legit_path))
    legit_urls, bad_urls = pb_input.read_dataset_url(legit_path, download_url_flag)
    bad_url_list.extend(bad_urls)
    legit_features, legit_corpus = extract_url_features(legit_urls, bad_url_list)

    print("Extracting Features from {}".format(phish_path))
    phish_urls, bad_urls = pb_input.read_dataset_url(phish_path, download_url_flag)
    bad_url_list.extend(bad_urls)
    phish_features, phish_corpus = extract_url_features(phish_urls, bad_url_list)

    features = legit_features + phish_features
    labels = ([0] * len(legit_features)) + ([1] * len(phish_features))
    corpus = legit_corpus + phish_corpus

    return features, labels, corpus


def extract_url_features(urls: List[URLData], bad_url_list):
    feature_list_dict = list()

    alexa_data = {}
    if phishbench_globals.config['URL_Feature_Types'].getboolean("HTML") and \
            phishbench_globals.config["HTML_Features"].getboolean("ranked_matrix"):
        alexa_path = phishbench_globals.config["Support Files"]["path_alexa_data"]
        alexa_data = read_alexa(alexa_path)

    corpus = []
    for url in tqdm(urls):
        feature_values, extraction_times = url_features(url, corpus, alexa_data, bad_url_list)
        feature_list_dict.append(feature_values)

    return feature_list_dict, corpus


def url_features(url: URLData, corpus, alexa_data, list_bad_urls):
    dict_feature_values = {}
    dict_extraction_times = {}
    try:
        phishbench_globals.logger.debug("rawurl: %s", str(url))
        phishbench_globals.summary.write("URL: {}".format(url))

        feature_types = phishbench_globals.config['URL_Feature_Types']
        if feature_types.getboolean('URL'):
            single_url_feature(url.raw_url, dict_feature_values, dict_extraction_times)
            phishbench_globals.logger.debug("url_features >>>>>> complete")
        if feature_types.getboolean("Network"):
            single_network_features(url, dict_feature_values, dict_extraction_times)
            phishbench_globals.logger.debug("network_features >>>>>> complete")
        if feature_types.getboolean("HTML"):
            single_url_html_features(url, alexa_data, dict_feature_values, dict_extraction_times)
            phishbench_globals.logger.debug("html_features >>>>>> complete")
            downloaded_website = url.downloaded_website
            soup = BeautifulSoup(downloaded_website.html, 'html5lib')
            if feature_types.getboolean("JavaScript"):
                single_javascript_features(soup, downloaded_website, dict_feature_values, dict_extraction_times)
                phishbench_globals.logger.debug("javascript features >>>>>> complete")
            corpus.append(str(soup))

    except Exception as e:
        phishbench_globals.logger.warning(traceback.format_exc())
        phishbench_globals.logger.exception(e)
        phishbench_globals.logger.warning(
            "This URL has trouble being extracted and "
            "will not be considered for further processing: %s", str(url))
        list_bad_urls.append(str(url))
    return dict_feature_values, dict_extraction_times


def single_url_feature(raw_url, list_features, list_time):
    phishbench_globals.logger.debug("Extracting single url features from %s", raw_url)

    Features.URL_url_length(raw_url, list_features, list_time)

    Features.URL_domain_length(raw_url, list_features, list_time)

    Features.URL_char_distance(raw_url, list_features, list_time)

    Features.URL_kolmogorov_shmirnov(raw_url, list_features, list_time)

    Features.URL_Kullback_Leibler_Divergence(raw_url, list_features, list_time)

    Features.URL_english_frequency_distance(raw_url, list_features, list_time)

    Features.URL_num_punctuation(raw_url, list_features, list_time)

    Features.URL_has_port(raw_url, list_features, list_time)

    Features.URL_has_https(raw_url, list_features, list_time)

    Features.URL_number_of_digits(raw_url, list_features, list_time)

    Features.URL_number_of_dots(raw_url, list_features, list_time)

    Features.URL_number_of_slashes(raw_url, list_features, list_time)

    Features.URL_consecutive_numbers(raw_url, list_features, list_time)

    Features.URL_special_char_count(raw_url, list_features, list_time)

    Features.URL_special_pattern(raw_url, list_features, list_time)

    Features.URL_Top_level_domain(raw_url, list_features, list_time)

    Features.URL_is_common_TLD(raw_url, list_features, list_time)

    Features.URL_number_of_dashes(raw_url, list_features, list_time)

    Features.URL_Http_middle_of_URL(raw_url, list_features, list_time)

    Features.URL_Has_More_than_3_dots(raw_url, list_features, list_time)

    Features.URL_Has_at_symbole(raw_url, list_features, list_time)

    Features.URL_Has_anchor_tag(raw_url, list_features, list_time)

    Features.URL_Null_in_Domain(raw_url, list_features, list_time)

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
    raw_url = url.raw_url
    downloaded_website = url.downloaded_website
    soup = BeautifulSoup(downloaded_website.html, 'html5lib')

    phishbench_globals.logger.debug("Extracting single html features from %s", raw_url)

    Features.HTML_ranked_matrix(soup, raw_url, alexa_data, list_features, list_time)

    Features.HTML_LTree_Features(soup, raw_url, list_features, list_time)

    Features.HTML_number_of_tags(soup, list_features, list_time)

    Features.HTML_number_of_head(soup, list_features, list_time)

    Features.HTML_number_of_html(soup, list_features, list_time)

    Features.HTML_number_of_body(soup, list_features, list_time)

    Features.HTML_number_of_titles(soup, list_features, list_time)

    Features.HTML_number_suspicious_content(soup, list_features, list_time)

    Features.HTML_number_of_iframes(soup, list_features, list_time)

    Features.HTML_number_of_input(soup, list_features, list_time)

    Features.HTML_number_of_img(soup, list_features, list_time)

    Features.HTML_number_of_tags(soup, list_features, list_time)

    Features.HTML_number_of_scripts(soup, list_features, list_time)

    Features.HTML_number_of_anchor(soup, list_features, list_time)

    Features.HTML_number_of_video(soup, list_features, list_time)

    Features.HTML_number_of_audio(soup, list_features, list_time)

    Features.HTML_number_of_hidden_svg(soup, list_features, list_time)

    Features.HTML_number_of_hidden_input(soup, list_features, list_time)

    Features.HTML_number_of_hidden_iframe(soup, list_features, list_time)

    Features.HTML_number_of_hidden_div(soup, list_features, list_time)

    Features.HTML_number_of_hidden_object(soup, list_features, list_time)

    Features.HTML_number_of_hidden_iframe(soup, list_features, list_time)

    Features.HTML_inbound_count(soup, raw_url, list_features, list_time)

    Features.HTML_outbound_count(soup, raw_url, list_features, list_time)

    Features.HTML_inbound_href_count(soup, raw_url, list_features, list_time)

    Features.HTML_outbound_href_count(soup, raw_url, list_features, list_time)

    Features.HTML_Website_content_type(downloaded_website, list_features, list_time)

    Features.HTML_content_length(downloaded_website, list_features, list_time)

    Features.HTML_x_powered_by(downloaded_website, list_features, list_time)

    Features.HTML_URL_Is_Redirect(downloaded_website, raw_url, list_features, list_time)

    Features.HTML_Is_Login(downloaded_website.html, raw_url, list_features, list_time)


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
