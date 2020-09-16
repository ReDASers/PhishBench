import time

from . import Tfidf
from .Features_Support import *
from .feature_extraction.reflection import FeatureType
from .utils import phishbench_globals


# pylint: skip-file


##### Email URL features
def Email_Number_Url(url_All, list_features, list_time):
    if phishbench_globals.config["Email_URL_Features"]["Number_Url"] == "True":
        start = time.time()
        try:
            list_features["Number_Url"] = len(url_All)
        except Exception as e:
            phishbench_globals.logger.warning("exception: " + str(e))
            list_features["Number_Url"] = -1
        end = time.time()
        ex_time = end - start
        list_time["Number_Url"] = ex_time


def Email_URL_Number_Diff_Domain(url_All, list_features, list_time):
    if phishbench_globals.config["Email_URL_Features"]["Number_Diff_Domain"] == "True":
        start = time.time()
        list_Domains = []
        try:
            for url in url_All:
                parsed_url = urlparse(url)
                domain = parsed_url.hostname
                list_Domains.append(domain)
                # if domain not in list_Domains:
                #    list_Domains.append(domain)
            list_features["Number_Diff_Domain"] = len(set(list_Domains))
        except Exception as e:
            phishbench_globals.logger.warning("exception: " + str(e))
            list_features["Number_Diff_Domain"] = -1
        end = time.time()
        ex_time = end - start
        list_time["Number_Diff_Domain"] = ex_time


def Email_URL_Number_Diff_Subdomain(url_All, list_features, list_time):
    if phishbench_globals.config["Email_URL_Features"]["Number_Diff_Subdomain"] == "True":
        start = time.time()
        list_Subdomains = []
        try:
            for url in url_All:
                parsed_url = urlparse(url)
                domain = parsed_url.hostname
                subdomain = domain.split('.')[0]
                list_Subdomains.append(subdomain)
                # if domain not in list_Domains:
                #    list_Domains.append(domain)
            list_features["Number_Diff_Subdomain"] = len(set(list_Subdomains))
        except Exception as e:
            phishbench_globals.logger.warning("exception: " + str(e))
            list_features["Number_Diff_Subdomain"] = -1
        end = time.time()
        ex_time = end - start
        list_time["Number_Diff_Subdomain"] = ex_time


def Email_URL_Number_link_at(url_All, list_features, list_time):
    if phishbench_globals.config["Email_URL_Features"]["Number_link_at"] == "True":
        start = time.time()
        count = 0
        try:
            for url in url_All:
                if "@" in url:
                    count += 1
                    list_features["Number_link_at"] = count
        except Exception  as e:
            phishbench_globals.logger.warning("exception: " + str(e))
            list_features["Number_link_at"] = -1
        end = time.time()
        ex_time = end - start
        list_time["Number_link_at"] = ex_time


def Email_URL_Number_link_sec_port(url_All, list_features, list_time):
    if phishbench_globals.config["Email_URL_Features"]["Number_link_sec_port"] == "True":
        start = time.time()
        count = 0
        try:
            for url in url_All:
                if "::443" in url:
                    count += 1
                    list_features["Number_link_sec_port"] = count
        except Exception  as e:
            phishbench_globals.logger.warning("exception: " + str(e))
            list_features["Number_link_sec_port"] = -1
        end = time.time()
        ex_time = end - start
        list_time["Number_link_sec_port"] = ex_time


def Email_Body_tfidf_emails(list_time):
    if phishbench_globals.config["Email_Body_Features"]["tfidf_emails"] == "True":
        start = time.time()
        Tfidf_matrix = Tfidf.tfidf_emails()
        end = time.time()
        ex_time = end - start
        list_time["tfidf_emails"] = ex_time
        return Tfidf_matrix


def Email_Header_Header_Tokenizer(list_time):
    if phishbench_globals.config["Email_Header_Features"]["Header_Tokenizer"] == "True":
        start = time.time()
        header_tokenizer = Tfidf.Header_Tokenizer()
        end = time.time()
        ex_time = end - start
        list_time["header_tokenizer"] = ex_time
        return header_tokenizer


def HTML_tfidf_websites(list_time):
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["tfidf_websites"] == "True":
        start = time.time()
        Tfidf_matrix = Tfidf.tfidf_websites()
        end = time.time()
        ex_time = end - start
        list_time["tfidf_websites"] = ex_time
        return Tfidf_matrix
