import ntpath
import os
import pickle
import re
import time
import traceback

from bs4 import BeautifulSoup

from ... import Features
from ...Features_Support import Cleaning, read_alexa
from ...input import input as pb_input
from ...input.url_input import URLData
from ...utils import Globals


def Extract_Features_Urls_Testing():
    start_time = time.time()
    Globals.logger.info(">>>>> Feature extraction: Testing Set")
    dataset_path_legit_test = Globals.config["Dataset Path"]["path_legitimate_testing"]
    dataset_path_phish_test = Globals.config["Dataset Path"]["path_phishing_testing"]
    feature_list_dict_test = []
    extraction_time_dict_test = []
    bad_url_list = []
    labels_legit_test, data_legit_test = extract_url_features(dataset_path_legit_test, feature_list_dict_test,
                                                              extraction_time_dict_test, bad_url_list)
    labels_all_test, data_phish_test = extract_url_features(dataset_path_phish_test, feature_list_dict_test,
                                                            extraction_time_dict_test, bad_url_list)
    Globals.logger.debug(">>>>> Feature extraction: Testing Set >>>>> Done ")
    Globals.logger.info(">>>>> Cleaning >>>>")
    Globals.logger.debug("feature_list_dict_test: %d", len(feature_list_dict_test))
    Cleaning(feature_list_dict_test)
    Globals.logger.debug(">>>>> Cleaning >>>>>> Done")
    # Globals.logger.info("Number of bad URLs in training dataset: {}".format(len(Bad_URLs_List)))
    labels_test = []
    for i in range(labels_legit_test):
        labels_test.append(0)
    for i in range(labels_all_test - labels_legit_test):
        labels_test.append(1)

    corpus_test = data_legit_test + data_phish_test
    Globals.logger.info("--- %.2f final count seconds ---", (time.time() - start_time))
    return feature_list_dict_test, labels_test, corpus_test


def Extract_Features_Urls_Training():
    # Globals.summary.open(Globals.config["Summary"]["Path"],'w')
    if Globals.config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
        Globals.logger.info("===============================================================")
        Globals.logger.info("===============================================================")
        Globals.logger.info(">>>>> Feature extraction: Training Set >>>>>")
        dataset_path_legit_train = Globals.config["Dataset Path"]["path_legitimate_training"]
        dataset_path_phish_train = Globals.config["Dataset Path"]["path_phishing_training"]
        feature_list_dict_train = []
        extraction_time_dict_train = []
        bad_url_list = []
        start_time = time.time()
        # with open("Data_Dump/URLs_Training/features_url_training_legit.pkl",'ab') as feature_tracking:
        labels_legit_train, data_legit_train = extract_url_features(dataset_path_legit_train, feature_list_dict_train,
                                                                    extraction_time_dict_train, bad_url_list)
        labels_all_train, data_phish_train = extract_url_features(dataset_path_phish_train, feature_list_dict_train,
                                                                  extraction_time_dict_train, bad_url_list)
        Globals.logger.info("Feature extraction time is: %ds", (time.time() - start_time))
        Globals.logger.debug(">>>>> Feature extraction: Training Set >>>>> Done ")
        Cleaning(feature_list_dict_train)
        Globals.logger.debug(">>>>> Cleaning >>>>>> Done")
        # Globals.logger.info("Number of bad URLs in training dataset: {}".format(len(Bad_URLs_List)))

        labels_train = []
        for i in range(labels_legit_train):
            labels_train.append(0)
        for i in range(labels_all_train - labels_legit_train):
            labels_train.append(1)

        # Globals.logger.info("\nfeature_list_dict_train2: {}\n".format(feature_list_dict_train2))
        corpus_train = data_legit_train + data_phish_train
        #
        #        #Globals.logger.info("--- %s final count seconds ---" % (time.time() - start_time))
        return feature_list_dict_train, labels_train, corpus_train
    return None, None, None


def extract_url_features(dataset_path, feature_list_dict, extraction_time_list_dict, bad_url_list):
    download_url_flag = Globals.config['Network_Features'].getboolean('network_features') or \
                        Globals.config['HTML_Features'].getboolean('HTML_features')
    url_list, bad_urls = pb_input.read_dataset_url(dataset_path, download_url_flag)
    bad_url_list.extend(bad_urls)
    alexa_data = {}
    if Globals.config['URL_Feature_Types'].getboolean("HTML") and \
            Globals.config["HTML_Features"].getboolean("ranked_matrix"):
        alexa_path = Globals.config["Support Files"]["path_alexa_data"]
        alexa_data = read_alexa(alexa_path)

    corpus = []
    for url in url_list:
        feature_values, extraction_times = url_features(url, corpus, alexa_data, bad_url_list)
        feature_list_dict.append(feature_values)
        extraction_time_list_dict.append(extraction_times)

        norm_path = ntpath.normpath(str(url))
        feature_dump_path = "Data_Dump/URLs_Backup/" + '_'.join(re.split(r'[:\\]+', norm_path))
        if not os.path.exists("Data_Dump/URLs_Backup"):
            os.makedirs("Data_Dump/URLs_Backup")
        dump_features(url, feature_values, extraction_times, feature_dump_path)
        for feature, extraction_time in extraction_times.items():
            Globals.summary.write("{} \n".format(feature))
            Globals.summary.write("extraction time: {} \n".format(extraction_time))
        Globals.summary.write("\n#######\n")

    count_files = len(feature_list_dict)
    return count_files, corpus


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
            html = url.downloaded_website.html
            soup = BeautifulSoup(html, 'html5lib')
            single_url_html_features(soup, html, url, alexa_data, dict_feature_values, dict_extraction_times)
            Globals.logger.debug("html_features >>>>>> complete")
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
    Features.URL_url_length(raw_url, list_features, list_time)
    Globals.logger.debug("url_length")
    Features.URL_domain_length(raw_url, list_features, list_time)
    Globals.logger.debug("domain_length")

    Features.URL_char_distance(raw_url, list_features, list_time)
    Globals.logger.debug("url_char_distance")

    Features.URL_kolmogorov_shmirnov(list_features, list_time)
    Globals.logger.debug("kolmogorov_shmirnov")

    Features.URL_Kullback_Leibler_Divergence(list_features, list_time)
    Globals.logger.debug("Kullback_Leibler_Divergence")

    Features.URL_english_frequency_distance(list_features, list_time)
    Globals.logger.debug("english_frequency_distance")

    Features.URL_num_punctuation(raw_url, list_features, list_time)
    Globals.logger.debug("num_punctuation")

    Features.URL_has_port(raw_url, list_features, list_time)
    Globals.logger.debug("has_port")

    Features.URL_has_https(raw_url, list_features, list_time)
    Globals.logger.debug("has_https")

    Features.URL_number_of_digits(raw_url, list_features, list_time)
    Globals.logger.debug("number_of_digits")

    Features.URL_number_of_dots(raw_url, list_features, list_time)
    Globals.logger.debug("number_of_dots")

    Features.URL_number_of_slashes(raw_url, list_features, list_time)
    Globals.logger.debug("number_of_slashes")

    Features.URL_digit_letter_ratio(raw_url, list_features, list_time)
    Globals.logger.debug("digit_letter_ratio")

    Features.URL_consecutive_numbers(raw_url, list_features, list_time)
    Globals.logger.debug("consecutive_numbers")

    Features.URL_special_char_count(raw_url, list_features, list_time)
    Globals.logger.debug("special_char_count")

    Features.URL_special_pattern(raw_url, list_features, list_time)
    Globals.logger.debug("special_pattern")

    Features.URL_Top_level_domain(raw_url, list_features, list_time)
    Globals.logger.debug("Top_level_domain")

    Features.URL_is_common_TLD(raw_url, list_features, list_time)
    Globals.logger.debug("is_common_TLD")

    Features.URL_number_of_dashes(raw_url, list_features, list_time)
    Globals.logger.debug('URL_number_of_dashes')

    Features.URL_Http_middle_of_URL(raw_url, list_features, list_time)
    Globals.logger.debug('URL_Http_middle_of_URL')

    Features.URL_Has_More_than_3_dots(raw_url, list_features, list_time)
    Globals.logger.debug('URL_Has_More_than_3_dots')

    Features.URL_Has_at_symbole(raw_url, list_features, list_time)
    Globals.logger.debug("URL_Has_at_symbole")

    Features.URL_Has_anchor_tag(raw_url, list_features, list_time)
    Globals.logger.debug("URL_Has_anchor_tag")

    Features.URL_Null_in_Domain(raw_url, list_features, list_time)
    Globals.logger.debug("URL_Null_in_Domain")

    Features.URL_Token_Count(raw_url, list_features, list_time)
    Globals.logger.debug("URL_Token_Count")

    Features.URL_Average_Path_Token_Length(raw_url, list_features, list_time)
    Globals.logger.debug("URL_Average_Path_Token_Length")

    Features.URL_Average_Domain_Token_Length(raw_url, list_features, list_time)
    Globals.logger.debug("URL_Average_Domain_Token_Length")

    Features.URL_Longest_Domain_Token(raw_url, list_features, list_time)
    Globals.logger.debug('URL_Longest_Domain_Token')

    Features.URL_Protocol_Port_Match(raw_url, list_features, list_time)
    Globals.logger.debug('URL_Protocol_Port_Match')

    Features.URL_Has_WWW_in_Middle(raw_url, list_features, list_time)
    Globals.logger.debug('URL_Has_WWW_in_Middle')

    Features.URL_Has_Hex_Characters(raw_url, list_features, list_time)
    Globals.logger.debug('URL_Has_Hex_Characters')

    Features.URL_Double_Slashes_Not_Beginning_Count(raw_url, list_features, list_time)
    Globals.logger.debug("URL_Double_Slashes_Not_Beginning_Count")

    Features.URL_Brand_In_Url(raw_url, list_features, list_time)
    Globals.logger.debug("URL_Bran_In_URL")

    Features.URL_Is_Whitelisted(raw_url, list_features, list_time)
    Globals.logger.debug("URL_Is_Whitelisted")


def single_url_html_features(soup, html, url, alexa_data, list_features, list_time):
    Features.HTML_ranked_matrix(soup, url, alexa_data, list_features, list_time)
    Globals.logger.debug("ranked_matrix")

    Features.HTML_LTree_Features(soup, url, list_features, list_time)
    Globals.logger.debug("LTree_Features")

    Features.HTML_number_of_tags(soup, list_features, list_time)
    Globals.logger.debug("number_of_tags")

    Features.HTML_number_of_head(soup, list_features, list_time)
    Globals.logger.debug("number_of_head")

    Features.HTML_number_of_html(soup, list_features, list_time)
    Globals.logger.debug("number_of_html")

    Features.HTML_number_of_body(soup, list_features, list_time)
    Globals.logger.debug("number_of_body")

    Features.HTML_number_of_titles(soup, list_features, list_time)
    Globals.logger.debug("number_of_titles")

    Features.HTML_number_suspicious_content(soup, list_features, list_time)
    Globals.logger.debug("number_suspicious_content")

    Features.HTML_number_of_iframes(soup, list_features, list_time)
    Globals.logger.debug("number_of_iframes")

    Features.HTML_number_of_input(soup, list_features, list_time)
    Globals.logger.debug("number_of_input")

    Features.HTML_number_of_img(soup, list_features, list_time)
    Globals.logger.debug("number_of_img")

    Features.HTML_number_of_tags(soup, list_features, list_time)
    Globals.logger.debug("number_of_tags")

    Features.HTML_number_of_scripts(soup, list_features, list_time)
    Globals.logger.debug("number_of_scripts")

    Features.HTML_number_of_anchor(soup, list_features, list_time)
    Globals.logger.debug("number_of_anchor")

    Features.HTML_number_of_video(soup, list_features, list_time)
    Globals.logger.debug("number_of_video")

    Features.HTML_number_of_audio(soup, list_features, list_time)
    Globals.logger.debug("number_of_audio")

    Features.HTML_number_of_hidden_iframe(soup, list_features, list_time)
    Globals.logger.debug("number_of_hidden_iframe")

    Features.HTML_number_of_hidden_div(soup, list_features, list_time)
    Globals.logger.debug("number_of_hidden_div")

    Features.HTML_number_of_hidden_object(soup, list_features, list_time)
    Globals.logger.debug("number_of_hidden_object")

    Features.HTML_number_of_hidden_iframe(soup, list_features, list_time)
    Globals.logger.debug("number_of_hidden_iframe")

    Features.HTML_inbound_count(soup, url, list_features, list_time)
    Globals.logger.debug("inbound_count")

    Features.HTML_outbound_count(soup, url, list_features, list_time)
    Globals.logger.debug("outbound_count")

    Features.HTML_inbound_href_count(soup, url, list_features, list_time)
    Globals.logger.debug("inbound_href_count")

    Features.HTML_outbound_href_count(soup, url, list_features, list_time)
    Globals.logger.debug("outbound_href_count")

    Features.HTML_Website_content_type(html, list_features, list_time)
    Globals.logger.debug("content_type")

    Features.HTML_content_length(html, list_features, list_time)
    Globals.logger.debug("content_length")

    Features.HTML_x_powered_by(html, list_features, list_time)
    Globals.logger.debug("x_powered_by")

    Features.HTML_URL_Is_Redirect(html, url, list_features, list_time)
    Globals.logger.debug("URL_Is_Redirect")

    Features.HTML_Is_Login(html.html, url, list_features, list_time)
    Globals.logger.debug("HTML_Is_Login")


def single_javascript_features(soup, html, list_features, list_time):
    Features.Javascript_number_of_exec(soup, list_features, list_time)
    Globals.logger.debug("number_of_exec")

    Features.Javascript_number_of_escape(soup, list_features, list_time)
    Globals.logger.debug("number_of_escape")

    Features.Javascript_number_of_eval(soup, list_features, list_time)
    Globals.logger.debug("number_of_eval")

    Features.Javascript_number_of_link(soup, list_features, list_time)
    Globals.logger.debug("number_of_link")

    Features.Javascript_number_of_unescape(soup, list_features, list_time)
    Globals.logger.debug("number_of_unescape")

    Features.Javascript_number_of_search(soup, list_features, list_time)
    Globals.logger.debug("number_of_search")

    Features.Javascript_number_of_setTimeout(soup, list_features, list_time)
    Globals.logger.debug("number_of_setTimeout")

    Features.Javascript_number_of_iframes_in_script(soup, list_features, list_time)
    Globals.logger.debug("number_of_iframes_in_script")

    Features.Javascript_number_of_event_attachment(soup, list_features, list_time)
    Globals.logger.debug("number_of_event_attachment")

    Features.Javascript_rightclick_disabled(html, list_features, list_time)
    Globals.logger.debug("rightclick_disabled")

    Features.Javascript_number_of_total_suspicious_features(list_features, list_time)
    Globals.logger.debug("number_of_total_suspicious_features")


def single_network_features(url, list_features, list_time):
    Features.Network_creation_date(url.domain_whois, list_features, list_time)
    Globals.logger.debug("creation_date")

    Features.Network_expiration_date(url.domain_whois, list_features, list_time)
    Globals.logger.debug("expiration_date")

    Features.Network_updated_date(url.whois_info, list_features, list_time)
    Globals.logger.debug("updated_date")

    Features.Network_as_number(url.ip_whois, list_features, list_time)
    Globals.logger.debug("as_number")

    Features.Network_number_name_server(url.dns_results, list_features, list_time)
    Globals.logger.debug("number_name_server")

    Features.Network_dns_ttl(url.raw_url, list_features, list_time)
    Globals.logger.debug("dns_ttl")

    Features.Network_DNS_Info_Exists(url.raw_url, list_features, list_time)
    Globals.logger.debug('DNS_Info_Exists')
