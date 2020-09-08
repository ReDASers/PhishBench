import statistics
import time
from datetime import datetime

import dns.resolver
import tldextract
from lxml import html as lxml_html

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
    if phishbench_globals.config[FeatureType.URL_WEBSITE_JAVASCRIPT.value][
        "number_of_total_suspicious_features"] == "True":
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
