import ntpath
import os
import pickle
import re
import time
import traceback

from bs4 import BeautifulSoup
from tqdm import tqdm

from ... import Features
from ...Features_Support import Cleaning, read_alexa
from ...input import input as pb_input
from ...input.url_input import URLData
from ...utils import Globals


def Extract_Features_Urls_Testing():
    print(">>>>> Feature extraction: Testing Set")

    dataset_path_legit_test = Globals.config["Dataset Path"]["path_legitimate_testing"]
    dataset_path_phish_test = Globals.config["Dataset Path"]["path_phishing_testing"]
    feature_list_dict_test = []
    extraction_time_dict_test = []
    bad_url_list = []

    num_legit, data_legit_test = extract_url_features(dataset_path_legit_test, feature_list_dict_test,
                                                      extraction_time_dict_test, bad_url_list)
    num_phish, data_phish_test = extract_url_features(dataset_path_phish_test, feature_list_dict_test,
                                                      extraction_time_dict_test, bad_url_list)
    Globals.logger.debug(">>>>> Feature extraction: Testing Set >>>>> Done ")
    print(">>>>> Cleaning >>>>")
    Globals.logger.debug("feature_list_dict_test: %d", len(feature_list_dict_test))
    Cleaning(feature_list_dict_test)
    print(">>>>> Cleaning >>>>>> Done")
    print("Number of bad URLs in training dataset: {}".format(len(bad_url_list)))

    labels_test = ([0] * num_legit) + ([1] * num_phish)

    corpus_test = data_legit_test + data_phish_test

    return feature_list_dict_test, labels_test, corpus_test


def Extract_Features_Urls_Training():
    # Globals.summary.open(Globals.config["Summary"]["Path"],'w')
    if Globals.config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
        print("===============================================================")
        print("===============================================================")
        print(">>>>> Feature extraction: Training Set >>>>>")

        dataset_path_legit_train = Globals.config["Dataset Path"]["path_legitimate_training"]
        dataset_path_phish_train = Globals.config["Dataset Path"]["path_phishing_training"]
        feature_list_dict_train = []
        extraction_time_dict_train = []
        bad_url_list = []

        num_legit, data_legit_train = extract_url_features(dataset_path_legit_train, feature_list_dict_train,
                                                           extraction_time_dict_train, bad_url_list)
        num_phish, data_phish_train = extract_url_features(dataset_path_phish_train, feature_list_dict_train,
                                                           extraction_time_dict_train, bad_url_list)

        print(">>>>> Feature extraction: Training Set >>>>> Done ")
        Cleaning(feature_list_dict_train)
        print(">>>>> Cleaning >>>>>> Done")
        print("Number of bad URLs in training dataset: {}".format(len(bad_url_list)))

        labels_train = ([0] * num_legit) + ([1] * num_phish)
        corpus_train = data_legit_train + data_phish_train

        return feature_list_dict_train, labels_train, corpus_train
    return None, None, None


def extract_url_features(dataset_path, feature_list_dict, extraction_time_list_dict, bad_url_list):
    download_url_flag = Globals.config['URL_Feature_Types'].getboolean('Network') or \
                        Globals.config['URL_Feature_Types'].getboolean('HTML')
    url_list, bad_urls = pb_input.read_dataset_url(dataset_path, download_url_flag)
    bad_url_list.extend(bad_urls)
    alexa_data = {}
    if Globals.config['URL_Feature_Types'].getboolean("HTML") and \
            Globals.config["HTML_Features"].getboolean("ranked_matrix"):
        alexa_path = Globals.config["Support Files"]["path_alexa_data"]
        alexa_data = read_alexa(alexa_path)

    corpus = []
    count_files = len(feature_list_dict)
    for url in tqdm(url_list):
        feature_values, extraction_times = url_features(url, corpus, alexa_data, bad_url_list)
        feature_list_dict.append(feature_values)
        extraction_time_list_dict.append(extraction_times)

        # This code causes PhishBench to crash with certain URLs.
        # See Issue #46 on GitHub
        # TODO: Rewrite in a way that doesn't cause PhishBench to crash due to invalid paths
        # norm_path = ntpath.normpath(str(url))
        # feature_dump_path = "Data_Dump/URLs_Backup/" + '_'.join(re.split(r'[:\\]+', norm_path))
        # if not os.path.exists("Data_Dump/URLs_Backup"):
        #     os.makedirs("Data_Dump/URLs_Backup")
        # dump_features(url, feature_values, extraction_times, feature_dump_path)
        for feature, extraction_time in extraction_times.items():
            Globals.summary.write("{} \n".format(feature))
            Globals.summary.write("extraction time: {} \n".format(extraction_time))
        Globals.summary.write("\n#######\n")

    return len(feature_list_dict) - count_files, corpus


def url_features(url: URLData, corpus, alexa_data, list_bad_urls):
    dict_feature_values = {}
    dict_extraction_times = {}
    try:
        Globals.logger.debug("rawurl: %s", str(url))
        Globals.summary.write("URL: {}".format(url))

        feature_types = Globals.config['URL_Feature_Types']
        if feature_types.getboolean('URL'):
            single_url_feature(url.raw_url, dict_feature_values, dict_extraction_times)
            Globals.logger.debug("url_features >>>>>> complete")
        if feature_types.getboolean("Network"):
            single_network_features(url, dict_feature_values, dict_extraction_times)
            Globals.logger.debug("network_features >>>>>> complete")
        if feature_types.getboolean("HTML"):
            single_url_html_features(url, alexa_data, dict_feature_values, dict_extraction_times)
            Globals.logger.debug("html_features >>>>>> complete")
            downloaded_website = url.downloaded_website
            soup = BeautifulSoup(downloaded_website.html, 'html5lib')
            if feature_types.getboolean("JavaScript"):
                single_javascript_features(soup, html, dict_feature_values, dict_extraction_times)
                Globals.logger.debug("javascript feautures >>>>>> complete")
            corpus.append(str(soup))

    except Exception as e:
        Globals.logger.warning(traceback.format_exc())
        Globals.logger.warning(e)
        Globals.logger.warning(
            "This URL has trouble being extracted and "
            "will not be considered for further processing: %s", str(url))
        list_bad_urls.append(str(url))
    return dict_feature_values, dict_extraction_times


def dump_features(url, feature_values, extraction_times, features_output_folder):
    Globals.logger.debug("list_features: %d", len(feature_values))
    raw_url = url.raw_url
    with open(features_output_folder + "_feature_vector.pkl", 'ab') as feature_tracking:
        pickle.dump("URL: " + raw_url, feature_tracking)
        pickle.dump(feature_values, feature_tracking)
    if Globals.config['HTML_Features'].getboolean('HTML_features'):
        with open(features_output_folder + "_html_content.pkl", 'ab') as feature_tracking:
            pickle.dump("URL: " + raw_url, feature_tracking)
            html = url.downloaded_website.html
            pickle.dump(html, feature_tracking)

    with open(features_output_folder + "_feature_vector.txt", 'a+') as f:
        f.write("URL: " + str(url) + '\n' + str(feature_values).replace('{', '').replace('}', '')
                .replace(': ', ':').replace(',', '') + '\n\n')
    with open(features_output_folder + "_time_stats.txt", 'a+') as f:
        f.write(
            "URL: " + str(url) + '\n' + str(extraction_times).replace('{', '').replace('}', '')
            .replace(': ', ':').replace(',', '') + '\n\n')


def single_url_feature(raw_url, list_features, list_time):
    Globals.logger.debug("Extracting single url features from %s", raw_url)

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

    Globals.logger.debug("Extracting single html features from %s", raw_url)

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
    Globals.logger.debug("Extracting single javascript features")

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
    Globals.logger.debug("Extracting network features from %S", url)
    Features.Network_creation_date(url.domain_whois, list_features, list_time)

    Features.Network_expiration_date(url.domain_whois, list_features, list_time)

    Features.Network_updated_date(url.domain_whois, list_features, list_time)

    Features.Network_as_number(url.ip_whois, list_features, list_time)

    Features.Network_number_name_server(url.dns_results, list_features, list_time)

    Features.Network_dns_ttl(url.raw_url, list_features, list_time)

    Features.Network_DNS_Info_Exists(url.raw_url, list_features, list_time)
