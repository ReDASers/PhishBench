import copy
import ntpath
import os
import pickle
import time
import traceback

from bs4 import BeautifulSoup

from . import Download_url
from ... import Features
from ...Features_Support import Cleaning, read_corpus, read_alexa
from ...utils import Globals
import requests

def Extract_Features_Urls_Testing():
    start_time = time.time()
    Globals.logger.info(">>>>> Feature extraction: Testing Set")
    dataset_path_legit_test = Globals.config["Dataset Path"]["path_legitimate_testing"]
    dataset_path_phish_test = Globals.config["Dataset Path"]["path_phishing_testing"]
    feature_list_dict_test = []
    extraction_time_dict_test = []
    Bad_URLs_List = []
    labels_legit_test, data_legit_test = extract_url_features(dataset_path_legit_test, feature_list_dict_test,
                                                              extraction_time_dict_test, Bad_URLs_List)
    labels_all_test, data_phish_test = extract_url_features(dataset_path_phish_test, feature_list_dict_test,
                                                            extraction_time_dict_test, Bad_URLs_List)
    Globals.logger.debug(">>>>> Feature extraction: Testing Set >>>>> Done ")
    Globals.logger.info(">>>>> Cleaning >>>>")
    Globals.logger.debug("feature_list_dict_test: {}".format(len(feature_list_dict_test)))
    Cleaning(feature_list_dict_test)
    Globals.logger.debug(">>>>> Cleaning >>>>>> Done")
    # Globals.logger.info("Number of bad URLs in training dataset: {}".format(len(Bad_URLs_List)))
    labels_test = []
    for i in range(labels_legit_test):
        labels_test.append(0)
    for i in range(labels_all_test - labels_legit_test):
        labels_test.append(1)

    corpus_test = data_legit_test + data_phish_test
    Globals.logger.info("--- %s final count seconds ---" % (time.time() - start_time))
    return feature_list_dict_test, labels_test, corpus_test


def Extract_Features_Urls_Training():
    # Globals.summary.open(Globals.config["Summary"]["Path"],'w')
    if Globals.config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
        start_time = time.time()
        Globals.logger.info("===============================================================")
        Globals.logger.info("===============================================================")
        Globals.logger.info(">>>>> Feature extraction: Training Set >>>>>")
        dataset_path_legit_train = Globals.config["Dataset Path"]["path_legitimate_training"]
        dataset_path_phish_train = Globals.config["Dataset Path"]["path_phishing_training"]
        feature_list_dict_train = []
        feature_list_dict_train2 = []
        extraction_time_dict_train = []
        Bad_URLs_List = []
        t0 = time.time()
        # with open("Data_Dump/URLs_Training/features_url_training_legit.pkl",'ab') as feature_tracking:
        labels_legit_train, data_legit_train = extract_url_features(dataset_path_legit_train, feature_list_dict_train,
                                                                    extraction_time_dict_train, Bad_URLs_List)
        labels_all_train, data_phish_train = extract_url_features(dataset_path_phish_train, feature_list_dict_train,
                                                                  extraction_time_dict_train, Bad_URLs_List)
        Globals.logger.info("Feature extraction time is: {}s".format(time.time() - t0))
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


def extract_url_features(dataset_path, feature_list_dict, extraction_time_dict, Bad_URLs_List):
    data = list()
    corpus_data = read_corpus(dataset_path)
    alexa_data = {}
    if Globals.config["HTML_Features"]["ranked_matrix"] == "True":
        alexa_path = Globals.config["Support Files"]["path_alexa_data"]
        alexa_data = read_alexa(alexa_path)
    data.extend(corpus_data)
    ## for debugging purposes, not used in the pipeline
    ###
    corpus = []
    dict_alexa_rank = {}
    for filepath in data:
        # path="Data_Dump/URLs_Backup/"+str(ntpath.normpath(filepath).split('\\'))
        # features_regex=re.compile(path+r"_features_?\d?.txt")
        # try:
        #     list_files=os.listdir('.')
        #     count_feature_files=len(re.findall(features_regex,''.join(list_files)))
        #     Globals.logger.debug(count_feature_files)
        #     features_output=path+"_feature_vector_"+str(count_feature_files+ 1)+".txt"
        # except Exception as e:
        #     features_output=path+"_feature_vector_error.txt"
        #     Globals.logger.warning("exception: " + str(e))
        dict_features = {}
        dict_time = {}
        Globals.logger.info("===================")
        Globals.logger.info(filepath)
        # with open("Data_Dump/URLs_Training/features_url_training_legit.pkl",'ab') as feature_tracking:
        url_features(filepath, dict_features, feature_list_dict, dict_time, extraction_time_dict, corpus, Bad_URLs_List, alexa_data)
        Globals.summary.write("filepath: {}\n\n".format(filepath))
        Globals.summary.write("features extracted for this file:\n")
        for feature in dict_time.keys():
            Globals.summary.write("{} \n".format(feature))
            Globals.summary.write("extraction time: {} \n".format(dict_time[feature]))
        Globals.summary.write("\n#######\n")
    count_files = len(feature_list_dict)
    return count_files, corpus


def url_features(filepath, list_features, list_dict, list_time, time_dict, corpus, Bad_URLs_List, alexa_data):
    times = []
    try:
        with open(filepath, 'r', encoding="ISO-8859-1") as f:
            for rawurl in f:
                rawurl = rawurl.strip().rstrip()
                try:
                    if not rawurl:
                        continue
                    Globals.logger.debug("rawurl:" + str(rawurl))
                    Globals.summary.write("URL: {}".format(rawurl))
                    t0 = time.time()
                    html, content, Error = Download_url.download_url(rawurl, list_time)
                    IPs, ipwhois, whois_output, domain = Download_url.extract_whois(html.url, list_time)
                    dns_lookup = Download_url.extract_dns_info(html.url, list_time)
                    if Error == 1:
                        Globals.logger.warning(
                            "This URL has trouble being extracted and will"
                            " not be considered for further processing:{}".format(rawurl))
                        Bad_URLs_List.append(rawurl)
                    else:
                        Globals.logger.debug("download_url >>>>>>>>> complete")
                        times.append(time.time() - t0)
                        # include https or http
                        url = rawurl.strip().rstrip('\n')
                        soup = BeautifulSoup(content, 'html5lib')  # content=html.text
                        single_url_html_features(soup, html, url, alexa_data, list_features, list_time)
                        single_url_feature(url, list_features, list_time)
                        Globals.logger.debug("html_featuers & url_features >>>>>> complete")
                        single_javascript_features(soup, html, list_features, list_time)
                        Globals.logger.debug("html_features & url_features & Javascript feautures >>>>>> complete")
                        single_network_features(dns_lookup, IPs, ipwhois, whois_output, url, list_features, list_time)
                        features_output = "Data_Dump/URLs_Backup/" + '_'.join(ntpath.normpath(filepath).split('\\'))
                        if not os.path.exists("Data_Dump/URLs_Backup"):
                            os.makedirs("Data_Dump/URLs_Backup")

                        dump_features(rawurl, str(soup), list_features, features_output, list_dict, list_time,
                                      time_dict)
                        corpus.append(str(soup))
                except Exception as e:
                    Globals.logger.warning(traceback.format_exc())
                    Globals.logger.warning(e)
                    Globals.logger.warning(
                        "This URL has trouble being extracted and "
                        "will not be considered for further processing:{}".format(rawurl))
                    Bad_URLs_List.append(rawurl)

    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        Globals.logger.debug(traceback.format_exc())
    Globals.logger.info("Download time is: {}".format(sum(times) / len(times)))


def dump_features(header, content, list_features, features_output, list_dict, list_time, time_dict):
    features_output = features_output.replace("..", "")
    Globals.logger.debug("list_features: " + str(len(list_features)))
    list_dict.append(copy.copy(list_features))
    time_dict.append(copy.copy(list_time))
    with open(features_output + "_feature_vector.pkl", 'ab') as feature_tracking:
        pickle.dump("URL: " + header, feature_tracking)
        pickle.dump(list_features, feature_tracking)
    with open(features_output + "_html_content.pkl", 'ab') as feature_tracking:
        pickle.dump("URL: " + header, feature_tracking)
        pickle.dump(content, feature_tracking)
    with open(features_output + "_feature_vector.txt", 'a+') as f:
        f.write("URL: " + str(header) + '\n' + str(list_features).replace('{', '').replace('}', '').replace(': ',
                                                                                                            ':').replace(
            ',', '') + '\n\n')
    with open(features_output + "_time_stats.txt", 'a+') as f:
        f.write(
            "URL: " + str(header) + '\n' + str(list_time).replace('{', '').replace('}', '').replace(': ', ':').replace(
                ',', '') + '\n\n')


def single_url_feature(url, list_features, list_time):
    if Globals.config["URL_Features"]["url_features"] == "True":
        Features.URL_url_length(url, list_features, list_time)
        Globals.logger.debug("url_length")

        Features.URL_domain_length(url, list_features, list_time)
        Globals.logger.debug("domain_length")

        Features.URL_char_distance(url, list_features, list_time)
        Globals.logger.debug("url_char_distance")

        Features.URL_kolmogorov_shmirnov(list_features, list_time)
        Globals.logger.debug("kolmogorov_shmirnov")

        Features.URL_Kullback_Leibler_Divergence(list_features, list_time)
        Globals.logger.debug("Kullback_Leibler_Divergence")

        Features.URL_english_frequency_distance(list_features, list_time)
        Globals.logger.debug("english_frequency_distance")

        Features.URL_num_punctuation(url, list_features, list_time)
        Globals.logger.debug("num_punctuation")

        Features.URL_has_port(url, list_features, list_time)
        Globals.logger.debug("has_port")

        Features.URL_has_https(url, list_features, list_time)
        Globals.logger.debug("has_https")

        Features.URL_number_of_digits(url, list_features, list_time)
        Globals.logger.debug("number_of_digits")

        Features.URL_number_of_dots(url, list_features, list_time)
        Globals.logger.debug("number_of_dots")

        Features.URL_number_of_slashes(url, list_features, list_time)
        Globals.logger.debug("number_of_slashes")

        Features.URL_digit_letter_ratio(url, list_features, list_time)
        Globals.logger.debug("digit_letter_ratio")

        Features.URL_consecutive_numbers(url, list_features, list_time)
        Globals.logger.debug("consecutive_numbers")

        Features.URL_special_char_count(url, list_features, list_time)
        Globals.logger.debug("special_char_count")

        Features.URL_special_pattern(url, list_features, list_time)
        Globals.logger.debug("special_pattern")

        Features.URL_Top_level_domain(url, list_features, list_time)
        Globals.logger.debug("Top_level_domain")

        Features.URL_is_common_TLD(url, list_features, list_time)
        Globals.logger.debug("is_common_TLD")

        Features.URL_number_of_dashes(url, list_features, list_time)
        Globals.logger.debug('URL_number_of_dashes')

        Features.URL_Http_middle_of_URL(url, list_features, list_time)
        Globals.logger.debug('URL_Http_middle_of_URL')

        Features.URL_Has_More_than_3_dots(url, list_features, list_time)
        Globals.logger.debug('URL_Has_More_than_3_dots')

        Features.URL_Has_at_symbole(url, list_features, list_time)
        Globals.logger.debug("URL_Has_at_symbole")

        Features.URL_Has_anchor_tag(url, list_features, list_time)
        Globals.logger.debug("URL_Has_anchor_tag")

        Features.URL_Null_in_Domain(url, list_features, list_time)
        Globals.logger.debug("URL_Null_in_Domain")

        Features.URL_Token_Count(url, list_features, list_time)
        Globals.logger.debug("URL_Token_Count")

        Features.URL_Average_Path_Token_Length(url, list_features, list_time)
        Globals.logger.debug("URL_Average_Path_Token_Length")

        Features.URL_Average_Domain_Token_Length(url, list_features, list_time)
        Globals.logger.debug("URL_Average_Domain_Token_Length")

        Features.URL_Longest_Domain_Token(url, list_features, list_time)
        Globals.logger.debug('URL_Longest_Domain_Token')

        Features.URL_Protocol_Port_Match(url, list_features, list_time)
        Globals.logger.debug('URL_Protocol_Port_Match')

        Features.URL_Has_WWW_in_Middle(url, list_features, list_time)
        Globals.logger.debug('URL_Has_WWW_in_Middle')

        Features.URL_Has_Hex_Characters(url, list_features, list_time)
        Globals.logger.debug('URL_Has_Hex_Characters')

        Features.URL_Double_Slashes_Not_Beginning_Count(url, list_features, list_time)
        Globals.logger.debug("URL_Double_Slashes_Not_Beginning_Count")

        Features.URL_Brand_In_Url(url, list_features, list_time)
        Globals.logger.debug("URL_Bran_In_URL")

        Features.URL_Is_Whitelisted(url, list_features, list_time)
        Globals.logger.debug("URL_Is_Whitelisted")


def single_url_html_features(soup, html, url, alexa_data, list_features, list_time):
    if Globals.config["HTML_Features"]["html_features"] == "True":
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

        Features.HTML_number_of_hidden_svg(soup, list_features, list_time)
        Globals.logger.debug("number_of_hidden_svg")

        Features.HTML_number_of_hidden_input(soup, list_features, list_time)
        Globals.logger.debug("number_of_hidden_input")

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
    if Globals.config["HTML_Features"]["HTML_features"] == "True" and Globals.config["Javascript_Features"][
        "javascript_features"] == "True":
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


def single_network_features(dns_info, IPS, IP_whois, whois_info, url, list_features, list_time):
    if Globals.config["Network_Features"]["network_features"] == "True":
        Features.Network_creation_date(whois_info, list_features, list_time)
        Globals.logger.debug("creation_date")

        Features.Network_expiration_date(whois_info, list_features, list_time)
        Globals.logger.debug("expiration_date")

        Features.Network_updated_date(whois_info, list_features, list_time)
        Globals.logger.debug("updated_date")

        Features.Network_as_number(IP_whois, list_features, list_time)
        Globals.logger.debug("as_number")

        Features.Network_number_name_server(dns_info, list_features, list_time)
        Globals.logger.debug("number_name_server")

        Features.Network_dns_ttl(url, list_features, list_time)
        Globals.logger.debug("dns_ttl")

        Features.Network_DNS_Info_Exists(url, list_features, list_time)
        Globals.logger.debug('DNS_Info_Exists')
