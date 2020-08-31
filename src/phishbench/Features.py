import statistics
import time
from datetime import datetime

import dns.resolver
import tldextract
from lxml import html as lxml_html

from . import Tfidf
from .Features_Support import *
from .utils import phishbench_globals
from .feature_extraction.reflection import FeatureType


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



# def html_in_body(body, list_features, list_time):
#    if Globals.config["Features"]["html_in_body"] == "True":
#        start=time.time()
#        Email_Body_html=re.compile(r'text/html', flags=re.IGNORECASE)
#        try:
#            html_in_body=int(bool(re.search(Email_Body_html, body)))
#        except Exception as e:
#            Globals.logger.warning("exception: " + str(e))
#            html_in_body=0
#        list_features["html_in_body"]=html_in_body
#        end=time.time()
#        ex_time=end-start
#        list_time["html_in_body"]=ex_time
#        #list_features[""]=


# def  bodyTextNotSimSubjectAndMinOneLink()
# def Email_Body_body_num_func_words(body, list_features, list_time):
#   if Globals.config["Email_Body_Features"]["body_num_func_words"] == "True":
#       start=time.time()
#       body_num_func_words=0


# def body_unique_words() x
# def Email_Body_num_img_links() x
# def num_of_sub_domains() x
# def blacklist_words_in_subject() x


# source for style metrics: https://pypi.python.org/pypi/textstat

############################ HTML features


# START - ranked_matrix
def HTML_ranked_matrix(soup, url, alexa_data, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["ranked_matrix"] == "True":
        start = time.time()
        domain = url.split("//")[-1].split("/")[0]
        mean_and_sd = [0, 0]
        if soup:
            try:
                # get links from content
                all_redirectable_links = []
                link = tree_get_links(soup, 'link', 'href', '')
                link += tree_get_links(soup, 'img', 'src', '')
                link += tree_get_links(soup, 'video', 'src', '')
                link += tree_get_links(soup, 'a', 'src', '')
                link += tree_get_links(soup, 'a', 'href', '')
                link += tree_get_links(soup, 'meta', 'content', '/')
                link += tree_get_links(soup, 'script', 'src', '')
                for l in link:
                    if l.startswith("http"):
                        all_redirectable_links.append(l)
                # extract features: size, mean, standard deviation
                mean_and_sd = extract_features_ranked_matrix(all_redirectable_links, alexa_data, domain)
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                # make all values to -1
                mean_and_sd = [-1, -1]
        else:
            # make all values to 0
            phishbench_globals.logger.warning("empty soup")
        print(mean_and_sd)
        list_features["ranked_matrix_mean"] = mean_and_sd[0]
        list_features["ranked_matrix_sd"] = mean_and_sd[1]
        end = time.time()
        ex_time = end - start
        list_time["ranked_matrix"] = ex_time


def extract_features_ranked_matrix(links, alexa_data, original_domain):
    results = []
    for link in links:
        domain = link.split("//")[-1].split("/")[0]
        if domain.count(".") > 1:
            domain = domain.split(".")[-2] + "." + domain.split(".")[-1]
            results.append(get_rank(domain, alexa_data))
    mean = round(sum(results) / len(results), 2)
    original_rank = get_rank(original_domain, alexa_data)
    sd = round(statistics.stdev(results, xbar=original_rank), 2)
    return [mean, sd]


def get_rank(domain, alexa_data):
    if domain in alexa_data:
        alexa_rank = int(alexa_data[domain])
        if alexa_rank < 1000:
            alexa_rank = 1
        elif alexa_rank < 10000:
            alexa_rank = 2
        elif alexa_rank < 100000:
            alexa_rank = 3
        elif alexa_rank < 500000:
            alexa_rank = 4
        elif alexa_rank < 1000000:
            alexa_rank = 5
        elif alexa_rank < 5000000:
            alexa_rank = 6
        else:
            alexa_rank = 7
    else:
        alexa_rank = 8

    return alexa_rank


# END - ranked_matrix

# START - LTree features
def HTML_LTree_Features(soup, url, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["LTree_Features"] == "True":
        start = time.time()
        domain = url.split("//")[-1].split("/")[0]
        link_features = img_features = video_features = a_features = meta_features = script_features = [[0, 0, 0],
                                                                                                        [0, 0, 0],
                                                                                                        [0, 0, 0],
                                                                                                        [0, 0, 0],
                                                                                                        [0, 0, 0]]
        if soup:
            try:
                # get links from content
                link_link = tree_get_links(soup, 'link', 'href', '')
                img_link = tree_get_links(soup, 'img', 'src', '')
                video_link = tree_get_links(soup, 'video', 'src', '')
                a_link = tree_get_links(soup, 'a', 'src', '')
                a_link += tree_get_links(soup, 'a', 'href', '')
                meta_link = tree_get_links(soup, 'meta', 'content', '/')
                script_link = tree_get_links(soup, 'script', 'src', '')
                # extract features: size, mean, standard deviation
                link_features = extract_tree_features(link_link, domain)
                img_features = extract_tree_features(img_link, domain)
                video_features = extract_tree_features(video_link, domain)
                a_features = extract_tree_features(a_link, domain)
                meta_features = extract_tree_features(meta_link, domain)
                script_features = extract_tree_features(script_link, domain)
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                # make all values to -1
                link_features = img_features = video_features = a_features = meta_features = script_features \
                    = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
        else:
            # make all values to 0
            phishbench_globals.logger.warning("empty soup")
        add_features(list_features, link_features, 'link')
        add_features(list_features, img_features, 'img')
        add_features(list_features, video_features, 'video')
        add_features(list_features, a_features, 'a')
        add_features(list_features, meta_features, 'meta')
        add_features(list_features, script_features, 'script')
        end = time.time()
        ex_time = end - start
        list_time["LTree_features"] = ex_time


def tree_get_links(soup, tag, source, identifier):
    links = []
    for link in soup.findAll(tag):
        content = link.get(source)
        if content is not None:
            if identifier in content:
                links.append(content)
    return links


# Get size, mean, SD for a set of links
def extract_tree_features(links, domain):
    # TODO could use a input file for the list, not hardcoded!
    social_list = ['google.com', 'facebook.com', 'twitter.com', 'pinterest.com', 'instagram.com']
    features = []
    t1 = t2 = t3 = t4 = t5 = []
    for link in links:
        link_domain = link.split("//")[-1].split("/")[0]
        if domain in link:
            t1.append(link)
        elif link_domain in social_list:
            t2.append(link)
        elif "https:" == link[:6]:
            t3.append(link)
        elif "http:" == link[:5]:
            t4.append(link)
        else:
            t5.append(link)
    features.append(extract_by_type(t1))
    features.append(extract_by_type(t2))
    features.append(extract_by_type(t3))
    features.append(extract_by_type(t4))
    features.append(extract_by_type(t5))
    return features


def extract_by_type(link_list):
    block_size = len(link_list)
    block_mean = 0
    block_std = 0
    links_length = []
    for link in link_list:
        links_length.append(len(link))
    if len(links_length) > 0:
        block_mean = round(statistics.mean(links_length), 2)
    if len(links_length) > 1:
        block_std = round(statistics.stdev(links_length), 2)
    return [block_size, block_mean, block_std]


def add_features(list_features, features, tag):
    # features = [[3],[3],[3],[3],[3]], ..., [[],[],[],[],[]]
    # T1 - T5: 5 * 3
    name_list = ['size', 'mean', 'SD']
    for i, set_f in enumerate(features):
        # size mean, SD
        name_1 = "LTree_feature_" + str(tag) + "_T" + str(i) + "_"
        for j, f in enumerate(set_f):
            feature_name = name_1 + name_list[j]
            list_features[feature_name] = f


# END LTree features

def HTML_number_of_tags(soup, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_tags"] == "True":
        start = time.time()
        number_of_tags = 0
        if soup:
            try:
                all_tags = soup.find_all()
                number_of_tags = len(all_tags)
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_tags = -1
        else:
            number_of_tags = 0
        list_features["number_of_tags"] = number_of_tags
        end = time.time()
        ex_time = end - start
        list_time["number_of_tags"] = ex_time


def HTML_number_of_head(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_head"] == "True":
        start = time.time()
        number_of_head = 0
        if soup:
            try:
                number_of_head = len(soup.find_all('head'))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_head = -1
        list_features["number_of_head"] = number_of_head
        end = time.time()
        ex_time = end - start
        list_time["number_of_head"] = ex_time


def HTML_number_of_html(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_html"] == "True":
        start = time.time()
        number_of_html = 0
        if soup:
            try:
                number_of_html = len(soup.find_all('html'))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_html = -1
        list_features["number_of_html"] = number_of_html
        end = time.time()
        ex_time = end - start
        list_time["number_of_html"] = ex_time


def HTML_number_of_body(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_body"] == "True":
        start = time.time()
        number_of_body = 0
        if soup:
            try:
                number_of_body = len(soup.find_all('body'))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_body = -1
        list_features["number_of_body"] = number_of_body
        end = time.time()
        ex_time = end - start
        list_time["number_of_body"] = ex_time


def HTML_number_of_titles(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_titles"] == "True":
        start = time.time()
        number_of_titles = 0
        if soup:
            try:
                number_of_titles = len(soup.find_all('title'))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_titles = -1
        list_features["number_of_titles"] = number_of_titles
        end = time.time()
        ex_time = end - start
        list_time["number_of_titles"] = ex_time


def HTML_number_suspicious_content(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_suspicious_content"] == "True":
        start = time.time()
        all_tags = soup.find_all()
        number_suspicious_content = 0
        if soup:
            try:
                for tag in all_tags:
                    str_tag = str(tag)
                    if len(str_tag) > 128 and (str_tag.count(' ') / len(str_tag) < 0.05):
                        number_suspicious_content = number_suspicious_content + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_suspicious_content = -1
        list_features["number_suspicious_content"] = number_suspicious_content
        end = time.time()
        ex_time = end - start
        list_time["number_suspicious_content"] = ex_time


def HTML_number_of_iframes(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_iframes"] == "True":
        start = time.time()
        number_of_iframes = 0
        if soup:
            try:
                iframe_tags = soup.find_all('iframe')
                number_of_iframes = len(iframe_tags)
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_iframes = -1
        list_features["number_of_iframes"] = number_of_iframes
        end = time.time()
        ex_time = end - start
        list_time["number_of_iframes"] = ex_time


def HTML_number_of_input(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_input"] == "True":
        start = time.time()
        number_of_input = 0
        if soup:
            try:
                number_of_input = len(soup.find_all('input'))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_input = -1
        list_features["number_of_input"] = number_of_input
        end = time.time()
        ex_time = end - start
        list_time["number_of_input"] = ex_time


def HTML_number_of_img(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_img"] == "True":
        start = time.time()
        number_of_img = 0
        if soup:
            try:
                number_of_img = len(soup.find_all('img'))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_img = -1
        list_features["number_of_img"] = number_of_img
        end = time.time()
        ex_time = end - start
        list_time["number_of_img"] = ex_time


def HTML_number_of_scripts(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_scripts"] == "True":
        start = time.time()
        number_of_scripts = 0
        if soup:
            try:
                number_of_scripts = len(soup.find_all('script'))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_scripts = -1
        list_features["number_of_scripts"] = number_of_scripts
        end = time.time()
        ex_time = end - start
        list_time["number_of_scripts"] = ex_time


def HTML_number_of_anchor(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_anchor"] == "True":
        start = time.time()
        number_of_anchor = 0
        if soup:
            try:
                number_of_anchor = len(soup.find_all('a'))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_anchor = -1
        list_features["number_of_anchor"] = number_of_anchor
        end = time.time()
        ex_time = end - start
        list_time["number_of_anchor"] = ex_time


def HTML_number_of_embed(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_embed"] == "True":
        start = time.time()
        number_of_embed = 0
        if soup:
            try:
                number_of_embed = len(soup.find_all('embed'))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_embed = -1
        list_features["number_of_embed"] = number_of_embed
        end = time.time()
        ex_time = end - start
        list_time["number_of_embed"] = ex_time


def HTML_number_object_tags(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_object_tags"] == "True":
        start = time.time()
        number_object_tags = 0
        if soup:
            try:
                object_tags = len(soup.find_all('object'))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_object_tags = -1
        list_features["number_object_tags"] = number_object_tags
        end = time.time()
        ex_time = end - start
        list_time["number_object_tags"] = ex_time


def HTML_number_of_video(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_video"] == "True":
        start = time.time()
        number_of_video = 0
        if soup:
            try:
                number_of_video = len(soup.find_all('video'))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_video = -1
        list_features["number_of_video"] = number_of_video
        end = time.time()
        ex_time = end - start
        list_time["number_of_video"] = ex_time


def HTML_number_of_audio(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_audio"] == "True":
        start = time.time()
        number_of_audio = 0
        if soup:
            try:
                number_of_audio = len(soup.find_all('audio'))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_audio = -1
        list_features["number_of_audio"] = number_of_audio
        end = time.time()
        ex_time = end - start
        list_time["number_of_audio"] = ex_time


def HTML_number_of_hidden_input(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_hidden_input"] == "True":
        start = time.time()
        number_of_hidden_input = 0
        if soup:
            try:
                iframe_tags = soup.find_all('input')
                for tag in iframe_tags:
                    if tag.get('type') == "hidden":
                        number_of_hidden_input += 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_hidden_input = -1
        list_features["number_of_hidden_input"] = number_of_hidden_input
        end = time.time()
        ex_time = end - start
        list_time["number_of_hidden_input"] = ex_time


def HTML_number_of_hidden_svg(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_hidden_svg"] == "True":
        start = time.time()
        number_of_hidden_svg = 0
        if soup:
            try:
                iframe_tags = soup.find_all('svg')
                for tag in iframe_tags:
                    if tag.get('aria-hidden') == "true":
                        number_of_hidden_svg += 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_hidden_input = -1
        list_features["number_of_hidden_svg"] = number_of_hidden_svg
        end = time.time()
        ex_time = end - start
        list_time["number_of_hidden_svg"] = ex_time


def HTML_number_of_hidden_iframe(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_hidden_iframe"] == "True":
        start = time.time()
        number_of_hidden_iframe = 0
        if soup:
            try:
                iframe_tags = soup.find_all('iframe')
                for tag in iframe_tags:
                    if tag.get('height') == 0 or tag.get('width') == 0:
                        number_of_hidden_iframe = number_of_hidden_iframe + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_hidden_iframe = -1
        list_features["number_of_hidden_iframe"] = number_of_hidden_iframe
        end = time.time()
        ex_time = end - start
        list_time["number_of_hidden_iframe"] = ex_time


def HTML_number_of_hidden_div(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_hidden_div"] == "True":
        start = time.time()
        number_of_hidden_div = 0
        if soup:
            try:
                tags = soup.find_all('div')
                for tag in tags:
                    if tag.get('height') == 0 or tag.get('width') == 0:
                        number_of_hidden_div = number_of_hidden_div + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_hidden_div = -1
        list_features["number_of_hidden_div"] = number_of_hidden_div
        end = time.time()
        ex_time = end - start
        list_time["number_of_hidden_div"] = ex_time


def HTML_number_of_hidden_object(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["number_of_hidden_object"] == "True":
        start = time.time()
        number_of_hidden_object = 0
        if soup:
            try:
                object_tags = soup.find_all('object')
                for tag in object_tags:
                    if tag.get('height') == 0 or tag.get('width') == 0:
                        number_of_hidden_object = number_of_hidden_object + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_hidden_object = -1
        list_features["number_of_hidden_object"] = number_of_hidden_object
        end = time.time()
        ex_time = end - start
        list_time["number_of_hidden_object"] = ex_time


def HTML_inbound_count(soup, url, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["inbound_count"] == "True":
        start = time.time()
        inbound_count = 0
        if soup:
            try:
                url_extracted = tldextract.extract(url)
                local_domain = '{}.{}'.format(url_extracted.domain, url_extracted.suffix)
                tags = soup.find_all(['audio', 'embed', 'iframe', 'img', 'input', 'script', 'source', 'track', 'video'])
                for tag in tags:
                    src_address = tag.get('src')
                    if src_address != None:
                        if src_address.lower().startswith(("https", "http")):
                            extracted = tldextract.extract(src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link == local_domain:
                                inbound_count = inbound_count + 1
                        elif src_address.startswith("//"):
                            extracted = tldextract.extract("http:" + src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link == local_domain:
                                inbound_count = inbound_count + 1
                        else:
                            inbound_count = inbound_count + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                inbound_count = -1
        list_features["inbound_count"] = inbound_count
        end = time.time()
        ex_time = end - start
        list_time["inbound_count"] = ex_time


def HTML_outbound_count(soup, url, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["outbound_count"] == "True":
        start = time.time()
        outbound_count = 0
        if soup:
            try:
                url_extracted = tldextract.extract(url)
                local_domain = '{}.{}'.format(url_extracted.domain, url_extracted.suffix)
                tags = soup.find_all(['audio', 'embed', 'iframe', 'img', 'input', 'script', 'source', 'track', 'video'])
                for tag in tags:
                    src_address = tag.get('src')
                    if src_address != None:
                        if src_address.lower().startswith(("https", "http")):
                            extracted = tldextract.extract(src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link != local_domain:
                                outbound_count = outbound_count + 1
                        elif src_address.startswith("//"):
                            extracted = tldextract.extract("http:" + src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link != local_domain:
                                outbound_count = outbound_count + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                outbound_count = -1
        list_features["outbound_count"] = outbound_count
        end = time.time()
        ex_time = end - start
        list_time["outbound_count"] = ex_time


def HTML_inbound_href_count(soup, url, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["inbound_href_count"] == "True":
        start = time.time()
        inbound_href_count = 0
        if soup:
            try:
                url_extracted = tldextract.extract(url)
                local_domain = '{}.{}'.format(url_extracted.domain, url_extracted.suffix)
                tags = soup.find_all(['a', 'area', 'base', 'link'])
                for tag in tags:
                    src_address = tag.get('href')
                    if src_address is not None:
                        if src_address.lower().startswith(("https", "http")):
                            extracted = tldextract.extract(src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link == local_domain:
                                inbound_href_count = inbound_href_count + 1
                        elif src_address.startswith("//"):
                            extracted = tldextract.extract("http:" + src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link == local_domain:
                                inbound_href_count = inbound_href_count + 1
                        else:
                            inbound_href_count = inbound_href_count + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                inbound_href_count = -1
        list_features["inbound_href_count"] = inbound_href_count
        end = time.time()
        ex_time = end - start
        list_time["inbound_href_count"] = ex_time


def HTML_outbound_href_count(soup, url, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["outbound_href_count"] == "True":
        start = time.time()
        outbound_href_count = 0
        if soup:
            try:
                url_extracted = tldextract.extract(url)
                local_domain = '{}.{}'.format(url_extracted.domain, url_extracted.suffix)
                tags = soup.find_all(['a', 'area', 'base', 'link'])
                for tag in tags:
                    src_address = tag.get('href')
                    if src_address is not None:
                        if src_address.lower().startswith(("https", "http")):
                            extracted = tldextract.extract(src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link != local_domain:
                                outbound_href_count = outbound_href_count + 1
                        elif src_address.startswith("//"):
                            extracted = tldextract.extract("http:" + src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link != local_domain:
                                outbound_href_count = outbound_href_count + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                outbound_href_count = -1
        list_features["outbound_href_count"] = outbound_href_count
        end = time.time()
        ex_time = end - start
        list_time["outbound_href_count"] = ex_time


def HTML_content_length(html, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["content_length"] == "True":
        start = time.time()
        content_length = 0
        if html:
            try:
                if 'Content-Length' in html.headers:
                    content_length = html.headers['Content-Length']
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                content_length = -1
        list_features["content_length"] = int(content_length)
        end = time.time()
        ex_time = end - start
        list_time["content_length"] = ex_time


def HTML_x_powered_by(html, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["x_powered_by"] == "True":
        start = time.time()
        x_powered_by = ''
        if html:
            try:
                if 'X-Powered-By' in html.headers:
                    # x_powered_by = html.headers['X-Powered-By']
                    x_powered_by = html.headers["X-Powered-By"]
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                x_powered_by = "N/A"
        list_features["x_powered_by"] = x_powered_by
        end = time.time()
        ex_time = end - start
        list_time["x_powered_by"] = ex_time


def HTML_Is_Login(html, url, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_WEBSITE.value]["Is_Login"] == "True":
        start = time.time()
        userfield = passfield = emailfield = None
        _is_login = False
        doc = lxml_html.document_fromstring(html, base_url=url)
        try:
            form_element = doc.xpath('//form')
            if form_element:
                form = _pick_form(form_element)
            else:
                return _is_login
            for x in form.inputs:
                if not isinstance(x, html.InputElement):
                    continue
                type_ = x.type
                if type_ == 'password' and passfield is None:
                    passfield = x.name
                    _is_login = True
                    break
        except Exception as ex:
            _is_login = False

        list_features['is_login'] = _is_login
        end = time.time()
        ex_time = end - start
        list_time['is_login'] = ex_time


############################ URL features

##################################################################################
def URL_letter_occurrence(url, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_RAW.value]["letter_occurrence"] == "True":
        start = time.time()
        if url:
            ####
            try:
                parsed_url = urlparse(url)
                domain = '{uri.scheme}://{uri.hostname}/'.format(uri=parsed_url).lower()
                for x in string.ascii_lowercase:
                    list_features["letter_occurrence_" + x] = domain.count(x)
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                for x in range(26):
                    list_features["letter_occurrence_" + chr(x + ord('a'))] = -1
        else:
            for x in range(26):
                list_features["letter_occurrence_" + chr(x + ord('a'))] = 0
        end = time.time()
        ex_time = end - start
        list_time["letter_occurrence"] = ex_time
        # print("letter_occurrence >>>>>>>>>>>>>>>>>>: " + str(letter_occurrence))
        # list_features["letter_occurrence"]=letter_occurrence


def URL_consecutive_numbers(url, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_RAW.value]["consecutive_numbers"] == "True":
        start = time.time()
        result = 0
        if url:
            try:
                length = 0
                for c in url:
                    if c.isdigit():
                        length += 1
                    else:
                        result += length * length
                        length = 0
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                result = -1
        list_features["consecutive_numbers"] = result
        end = time.time()
        ex_time = end - start
        list_time["consecutive_numbers"] = ex_time


def URL_special_char_count(url, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_RAW.value]["special_char_count"] == "True":
        start = time.time()
        special_char_count = 0
        if url:
            try:
                special_char_count = url.count('@') + url.count('-')
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                special_char_count = -1
        list_features["special_char_count"] = special_char_count
        end = time.time()
        ex_time = end - start
        list_time["special_char_count"] = ex_time


def URL_Top_level_domain(url, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_RAW.value]["Top_level_domain"] == "True":
        start = time.time()
        tld = 0
        if url:
            try:
                extracted = tldextract.extract(url)
                tld = "{}".format(extracted.suffix)
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                tld = -1
        list_features["Top_level_domain"] = tld
        end = time.time()
        ex_time = end - start
        list_time["Top_level_domain"] = ex_time


# Devin's features
def URL_Has_More_than_3_dots(url, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_RAW.value]["Has_More_than_3_dots"] == "True":
        start = time.time()
        # regex_http=re.compile(r'')
        if url:
            try:
                url = url.replace('www.', '')
                count_dots = url.count('.')
                if count_dots >= 3:
                    list_features["Has_More_than_3_dots"] = 1
                else:
                    list_features["Has_More_than_3_dots"] = 0
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                list_features["Has_More_than_3_dots"] = -1
        else:
            list_features["Has_More_than_3_dots"] = 0
        end = time.time()
        ex_time = end - start
        list_time["Has_More_than_3_dots"] = ex_time


def URL_Has_anchor_tag(url, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_RAW.value]["Has_anchor_tag"] == "True":
        start = time.time()
        flag = 0
        if url:
            try:
                if '#' in url:
                    flag = 1
                else:
                    flag = 0
            except Exception as e:
                phishbench_globals.logger.warning("Exception: " + str(e))
                flag = -1
        list_features["Has_anchor_tag"] = flag
        end = time.time()
        ex_time = end - start
        list_time["Has_anchor_tag"] = ex_time


# PhishDef: URL Names Say It All
TOKEN_DELIMITER_REGEX = re.compile(r'[/\?\.=_&\-\']+')


def URL_Token_Count(url, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_RAW.value]["Token_Count"] == "True":
        start = time.time()
        count = 0
        if url:
            try:
                tokens = TOKEN_DELIMITER_REGEX.split(url)
                count = len(tokens)
            except Exception  as e:
                phishbench_globals.logger.warning("Exception: " + str(e))
                count = -1
        list_features["Token_Count"] = count
        end = time.time()
        ex_time = end - start
        list_time["Token_Count"] = ex_time


# Detecting Malicious URLs Using Lexical Analysis
def URL_Average_Path_Token_Length(url, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_RAW.value]["Average_Path_Token_Length"] == "True":
        start = time.time()
        average_token_length = 0
        if url:
            try:
                parsed_url = urlparse(url)
                path = '{uri.path}'.format(uri=parsed_url)
                list_tokens = TOKEN_DELIMITER_REGEX.split(path)
                list_len_tokens = [0 for x in range(len(list_tokens))]
                for token in list_tokens:
                    list_len_tokens[list_tokens.index(token)] = len(token)
                average_token_length = sum(list_len_tokens) / len(list_len_tokens)
            except Exception  as e:
                phishbench_globals.logger.warning("Exception: " + str(e))
                average_token_length = -1
        list_features["Average_Path_Token_Length"] = average_token_length
        end = time.time()
        ex_time = end - start
        list_time["Average_Path_Token_Length"] = ex_time


def URL_Average_Domain_Token_Length(url, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_RAW.value]["Average_Domain_Token_Length"] == "True":
        start = time.time()
        average_token_length = 0
        if url:
            try:
                parsed_url = urlparse(url)
                domain = '{uri.hostname}'.format(uri=parsed_url)
                list_len_tokens = []
                list_tokens = TOKEN_DELIMITER_REGEX.split(domain)
                for token in list_tokens:
                    list_len_tokens.append(len(token))
                average_token_length = sum(list_len_tokens) / len(list_len_tokens)
            except Exception  as e:
                phishbench_globals.logger.warning("Exception: " + str(e))
                average_token_length = -1
        list_features["Average_Domain_Token_Length"] = average_token_length
        end = time.time()
        ex_time = end - start
        list_time["Average_Domain_Token_Length"] = ex_time


def URL_Longest_Domain_Token(url, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_RAW.value]["Longest_Domain_Token"] == "True":
        start = time.time()
        try:
            if url == '':
                longest_token_len = 0
            else:
                parsed_url = urlparse(url)
                domain = '{uri.hostname}'.format(uri=parsed_url)
                list_tokens = TOKEN_DELIMITER_REGEX.split(domain)
                list_len_tokens = [len(x) for x in list_tokens]
                longest_token_len = max(list_len_tokens)
        except Exception as e:
            phishbench_globals.logger.warning("Exception: " + str(e))
            longest_token_len = -1
        list_features["Longest_Domain_Token"] = longest_token_len
        end = time.time()
        ex_time = end - start
        list_time["Longest_Domain_Token"] = ex_time


def URL_Protocol_Port_Match(url, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_RAW.value]["Protocol_Port_Match"] == "True":
        start = time.time()
        match = 1
        if url:
            try:
                parsed_url = urlparse(url)
                scheme = '{uri.scheme}'.format(uri=parsed_url).lower()
                port = '{uri.port}'.format(uri=parsed_url)
                protocol_port_list = [('http', 8080), ('http', 80), ('https', 443), ('ftp', 20), ('tcp', 20),
                                      ('scp', 20), ('ftp', 21), ('ssh', 22), ('telnet', 23), ('smtp', 25), ('dns', 53),
                                      ("pop3", 110), ("sftp", 115), ("imap", 143), ("smtp", 465), ("rlogin", 513),
                                      ("imap", 993), ("pop3", 995)]
                if port != 'None' and ((scheme, int(port)) not in protocol_port_list):
                    match = 0
                list_features["Protocol_Port_Match"] = match
            except Exception as e:
                phishbench_globals.logger.warning("Exception: {}".format(e))
                match = -1
        else:
            match = 0
        list_features["Protocol_Port_Match"] = match
        end = time.time()
        ex_time = end - start
        list_time["Protocol_Port_Match"] = ex_time


def URL_Has_WWW_in_Middle(url, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_RAW.value]["Has_WWW_in_Middle"] == "True":
        start = time.time()
        flag = 0
        # regex_www=re.compile(r'www')
        if url:
            try:
                parsed_url = urlparse(url)
                domain = '{uri.hostname}'.format(uri=parsed_url).lower()
                if 'www' in domain and domain.startswith('www') == False:
                    flag = 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                flag = -1
        list_features["Has_WWW_in_Middle"] = flag
        end = time.time()
        ex_time = end - start
        list_time["Has_WWW_in_Middle"] = ex_time


def URL_Has_Hex_Characters(url, list_features, list_time):
    if phishbench_globals.config['URL_Features']['Has_Hex_Characters'] == "True":
        start = time.time()
        flag = 0
        regex_hex = re.compile(r'%[1-9A-Z][1-9A-Z]')
        if url:
            try:
                # parsed_url = urlparse(url)
                # domain = '{uri.netloc}'.format(uri=parsed_url).lower()
                flag = int((bool(re.findall(regex_hex, url))))
            except Exception as e:
                phishbench_globals.logger.warning("Exception: {}".format(e))
                flag = -1
        list_features["Has_Hex_Characters"] = flag
        end = time.time()
        ex_time = end - start
        list_time["Has_Hex_Characters"] = ex_time


def URL_Double_Slashes_Not_Beginning_Count(url, list_features, list_time):
    if phishbench_globals.config['URL_Features']['Double_Slashes_Not_Beginning_Count'] == "True":
        start = time.time()
        flag = 0
        regex_2slashes = re.compile(r'//')
        if url:
            try:
                parsed_url = urlparse(url)
                path = '{uri.path}'.format(uri=parsed_url)
                flag = int((bool(re.findall(regex_2slashes, path))))
                list_features["Double_Slashes_Not_Beginning_Count"] = flag
            except Exception as e:
                phishbench_globals.logger.warning("Exception: {}".format(e))
                flag = -1
        list_features["Double_Slashes_Not_Beginning_Count"] = flag
        end = time.time()
        ex_time = end - start
        list_time["Double_Slashes_Not_Beginning_Count"] = ex_time


def URL_Brand_In_Url(url, list_features, list_time):
    if phishbench_globals.config['URL_Features']['Brand_In_Url'] == "True":
        start = time.time()
        tokens = re.split('[^a-zA-Z]', url)
        brands = ['microsoft', 'paypal', 'netflix', 'bankofamerica', 'wellsfargo', 'facebook', 'chase', 'orange', 'dhl',
                  'dropbox', 'docusign', 'adobe', 'linkedin', 'apple', 'google', 'banquepopulaire', 'alibaba',
                  'comcast', 'credit', 'agricole', 'yahoo', 'at', 'nbc', 'usaa', 'americanexpress', 'cibc', 'amazon',
                  'ing', 'bt']
        if any(token.lower() in brands for token in tokens):
            list_features["Brand_In_URL"] = 1
        else:
            list_features["Brand_In_URL"] = 0


def URL_Is_Whitelisted(url, list_features, list_time):
    if phishbench_globals.config['URL_Features']['Is_Whitelisted'] == "True":
        start = time.time()
        domain = tldextract.extract(url).domain
        whitelist = ['microsoft', 'paypal', 'netflix', 'bankofamerica', 'wellsfargo', 'facebook', 'chase', 'orange',
                     'dhl', 'dropbox', 'docusign', 'adobe', 'linkedin', 'apple', 'google', 'banquepopulaire', 'alibaba',
                     'comcast', 'credit', 'agricole', 'yahoo', 'at', 'nbc', 'usaa', 'americanexpress', 'cibc', 'amazon',
                     'ing', 'bt']
        if domain in whitelist:
            list_features["Is_Whitelisted"] = 1
        else:
            list_features["Is_Whitelisted"] = 0


# def URL_ foundURLProtocolAndPortDoNotMatch
############################ Network Features
# def registar_id(whois_info, registrar_mapping)
#   registar_id = 0
#  if 'registrar' in whois_info and whois_info['registrar'] in registrar_mapping:
#     registar_id = registrar_mapping[whois_info['registrar']]
# return registar_id


# def country(whois_info, list_features, list_time):
#    #global list_features
#    if Globals.config["Features"]["country"] == "True":
#        start=time.time()
#        country = "N/A"
#        try:
#            if 'country' in whois_info:
#                country = whois_info['country']
#        except Exception as e:
#            Globals.logger.warning("exception: " + str(e))
#        list_features["country"]=country
#        end=time.time()
#        ex_time=end-start
#        list_time["country"]=ex_time

# age of domain


def Network_creation_date(whois_info, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_NETWORK.value]["creation_date"] == "True":
        start = time.time()
        creation_date = 0.0
        if whois_info:
            try:
                if "creation_date" in whois_info:
                    dateTime = whois_info.get("creation_date")
                    if dateTime is not None:
                        if type(dateTime) is list:
                            creation_date = dateTime[0].timestamp()
                        elif type(dateTime) is str:
                            creation_date = datetime(year=1996, month=1, day=1).timestamp()
                        else:
                            creation_date = dateTime.timestamp()
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                creation_date = -1
        list_features["creation_date"] = creation_date
        end = time.time()
        ex_time = end - start
        list_time["creation_date"] = ex_time


def Network_expiration_date(whois_info, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_NETWORK.value]["expiration_date"] == "True":
        start = time.time()
        expiration_date = 0.0
        if whois_info:
            try:
                if "expiration_date" in whois_info:
                    dateTime = whois_info.get("expiration_date")
                    if dateTime is not None:
                        if type(dateTime) is list:
                            expiration_date = dateTime[0].timestamp()
                        elif type(dateTime) is str:
                            expiration_date = 0.0
                        else:
                            expiration_date = dateTime.timestamp()
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                expiration_date = -1
        list_features["expiration_date"] = expiration_date
        end = time.time()
        ex_time = end - start
        list_time["expiration_date"] = ex_time


def Network_updated_date(whois_info, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_NETWORK.value]["updated_date"] == "True":
        start = time.time()
        updated_date = 0.0
        if whois_info:
            try:
                if "updated_date" in whois_info:
                    update_date_field = whois_info["updated_date"]
                    if update_date_field is not None:
                        if type(update_date_field) is list:
                            updated_date = update_date_field[0].timestamp()
                        elif type(update_date_field) is datetime:
                            updated_date = update_date_field.timestamp()
                        else:
                            updated_date = 0.0
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                updated_date = -1
        list_features["updated_date"] = updated_date
        # print("----Update_date: {}".format(updated_date))
        end = time.time()
        ex_time = end - start
        list_time["updated_date"] = ex_time


def Network_as_number(IP_whois_list, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_NETWORK.value]["as_number"] == "True":
        start = time.time()
        as_number = 0
        if IP_whois_list:
            try:
                if 'asn' in IP_whois_list:
                    as_number = IP_whois_list['asn']
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                as_number = -1
        list_features["as_number"] = as_number
        end = time.time()
        ex_time = end - start
        list_time["as_number"] = ex_time


def Network_number_name_server(dns_info, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_NETWORK.value]["number_name_server"] == "True":
        start = time.time()
        number_name_server = 0
        if dns_info:
            try:
                if 'NS' in dns_info:
                    number_name_server = len(dns_info['NS'])
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_name_server = -1
        list_features["number_name_server"] = number_name_server
        end = time.time()
        ex_time = end - start
        list_time["number_name_server"] = ex_time


def Network_DNS_Info_Exists(url, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_NETWORK.value]["DNS_Info_Exists"] == "True":
        start = time.time()
        flag = 1
        if url:
            try:
                parsed_url = urlparse(url)
                domain = '{uri.hostname}'.format(uri=parsed_url)
                resolver = dns.resolver.Resolver()
                resolver.timeout = 3
                resolver.lifetime = 3
                try:
                    dns_info = resolver.query(domain, 'A')
                    flag = 1
                except (
                        dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers,
                        dns.resolver.Timeout) as e:
                    phishbench_globals.logger.warning("Exception: {}".format(e))
                    flag = 0
            except Exception as e:
                phishbench_globals.logger.warning("Exception: {}".format(e))
                flag = -1
                phishbench_globals.logger.debug(list_features["DNS_Info_Exists"])
        else:
            flag = 0
        list_features["DNS_Info_Exists"] = flag
        end = time.time()
        ex_time = end - start
        list_time["DNS_Info_Exists"] = ex_time


def Network_dns_ttl(url, list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_NETWORK.value]["dns_ttl"] == "True":
        start = time.time()
        dns_ttl = 0
        retry_count = 0
        if url:
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.hostname
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                dns_ttl = -1
            try:
                while True:
                    try:
                        dns_complete_info = dns.resolver.query(domain, 'A')
                        dns_ttl = dns_complete_info.rrset.ttl
                    except dns.exception.Timeout:
                        if retry_count > 3:
                            break
                        retry_count = retry_count + 1
                        continue
                    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers) as e:
                        phishbench_globals.logger.warning("Exception: {}".format(e))
                        dns_ttl = 0
                        break
                    break
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                dns_ttl = -1
        list_features["dns_ttl"] = dns_ttl
        end = time.time()
        ex_time = end - start
        list_time["dns_ttl"] = ex_time


############################ Javascript features
def Javascript_number_of_exec(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE_JAVASCRIPT.value]["number_of_exec"] == "True":
        start = time.time()
        number_of_exec = 0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'exec(' in script_text:
                            number_of_exec = number_of_exec + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_exec = -1
        list_features["number_of_exec"] = number_of_exec
        end = time.time()
        ex_time = end - start
        list_time["number_of_exec"] = ex_time


def Javascript_number_of_escape(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE_JAVASCRIPT.value]["number_of_escape"] == "True":
        start = time.time()
        number_of_escape = 0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'escape(' in script_text:
                            number_of_escape = number_of_escape + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_escape = -1
        list_features["number_of_escape"] = number_of_escape
        end = time.time()
        ex_time = end - start
        list_time["number_of_escape"] = ex_time


def Javascript_number_of_eval(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE_JAVASCRIPT.value]["number_of_eval"] == "True":
        start = time.time()
        number_of_eval = 0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'eval(' in script_text:
                            number_of_eval = number_of_eval + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_eval = -1
        list_features["number_of_eval"] = number_of_eval
        end = time.time()
        ex_time = end - start
        list_time["number_of_eval"] = ex_time


def Javascript_number_of_link(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE_JAVASCRIPT.value]["number_of_link"] == "True":
        start = time.time()
        number_of_link = 0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'link(' in script_text:
                            number_of_link = number_of_link + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_link = -1
        list_features["number_of_link"] = number_of_link
        end = time.time()
        ex_time = end - start
        list_time["number_of_link"] = ex_time


def Javascript_number_of_unescape(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE_JAVASCRIPT.value]["number_of_unescape"] == "True":
        start = time.time()
        number_of_unescape = 0
        scripts = soup.find_all('script')
        if soup:
            try:
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'unescape(' in script_text:
                            number_of_unescape = number_of_unescape + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_unescape = -1
        list_features["number_of_unescape"] = number_of_unescape
        end = time.time()
        ex_time = end - start
        list_time["number_of_unescape"] = ex_time


def Javascript_number_of_search(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE_JAVASCRIPT.value]["number_of_search"] == "True":
        start = time.time()
        number_of_search = 0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'search(' in script_text:
                            number_of_search = number_of_search + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_search = -1
        list_features["number_of_search"] = number_of_search
        end = time.time()
        ex_time = end - start
        list_time["number_of_search"] = ex_time


def Javascript_number_of_setTimeout(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE_JAVASCRIPT.value]["number_of_setTimeout"] == "True":
        start = time.time()
        number_of_setTimeout = 0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'setTimeout(' in script_text:
                            number_of_setTimeout = number_of_setTimeout + 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_setTimeout = -1
        list_features["number_of_setTimeout"] = number_of_setTimeout
        end = time.time()
        ex_time = end - start
        list_time["number_of_setTimeout"] = ex_time


def Javascript_number_of_iframes_in_script(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE_JAVASCRIPT.value]["number_of_iframes_in_script"] == "True":
        start = time.time()
        number_of_iframes_in_script = 0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        number_of_iframes_in_script = number_of_iframes_in_script + script_text.count("iframe")
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_iframes_in_script = -1
        list_features["number_of_iframes_in_script"] = number_of_iframes_in_script
        end = time.time()
        ex_time = end - start
        list_time["number_of_iframes_in_script"] = ex_time


def Javascript_number_of_event_attachment(soup, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE_JAVASCRIPT.value]["number_of_event_attachment"] == "True":
        start = time.time()
        number_of_event_attachment = 0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        number_of_event_attachment = number_of_event_attachment + len(re.findall(
                            "(?:addEventListener|attachEvent|dispatchEvent|fireEvent)\('(?:error|load|beforeunload|unload)'",
                            script_text.replace(" ", "")))
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                number_of_event_attachment = -1
        list_features["number_of_event_attachment"] = number_of_event_attachment
        end = time.time()
        ex_time = end - start
        list_time["number_of_event_attachment"] = ex_time


def Javascript_rightclick_disabled(html, list_features, list_time):
    # global list_features
    if phishbench_globals.config[FeatureType.URL_WEBSITE_JAVASCRIPT.value]["rightclick_disabled"] == "True":
        start = time.time()
        rightclick_disabled = 0
        if html:
            try:
                rightclick_disabled = 0
                # print(html.text.lower())
                if 'addEventListener(\'contextmenu\'' in html.html.lower():
                    rightclick_disabled = 1
            except Exception as e:
                phishbench_globals.logger.warning("exception: " + str(e))
                rightclick_disabled = -1
        list_features["rightclick_disabled"] = rightclick_disabled
        end = time.time()
        ex_time = end - start
        list_time["rightclick_disabled"] = ex_time


def Javascript_number_of_total_suspicious_features(list_features, list_time):
    if phishbench_globals.config[FeatureType.URL_WEBSITE_JAVASCRIPT.value]["number_of_total_suspicious_features"] == "True":
        start = time.time()
        number_of_total_suspicious_features = 0
        try:
            number_of_total_suspicious_features = list_features["number_of_exec"] + list_features["number_of_escape"] + \
                                                  list_features["number_of_eval"] + list_features["number_of_link"] + \
                                                  list_features["number_of_unescape"] + list_features[
                                                      "number_of_search"] \
                                                  + list_features["rightclick_disabled"] + list_features[
                                                      "number_of_event_attachment"] + list_features[
                                                      "number_of_iframes_in_script"] + list_features[
                                                      "number_of_event_attachment"] + list_features[
                                                      "number_of_setTimeout"]
        except Exception as e:
            phishbench_globals.logger.warning("exception: " + str(e))
            number_of_total_suspicious_features = -1
        list_features["number_of_total_suspicious_features"] = number_of_total_suspicious_features
        end = time.time()
        ex_time = end - start
        list_time["number_of_total_suspicious_features"] = ex_time


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
