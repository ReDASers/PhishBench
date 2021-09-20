"""
Link-Tree features from Phishing Sites Detection from a Web Developer’s Perspective
Using Machine Learning
"""
import csv
import os.path
import pathlib
import statistics

from bs4 import BeautifulSoup
from tldextract import tldextract

from ...reflection import FeatureType, register_feature
from ....input import URLData


@register_feature(FeatureType.URL_WEBSITE, 'link_alexa_global_rank')
def link_alexa_global_rank(url: URLData):
    """
    The mean and standard deviation of the global ranks normalized by ranges
    <1,000, <10,000, <100,000, <500,000, <1,000,000, and unranked
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')

    links = _tree_get_links(soup, 'link', 'href', '')
    links += _tree_get_links(soup, 'img', 'src', '')
    links += _tree_get_links(soup, 'video', 'src', '')
    links += _tree_get_links(soup, 'a', 'src', '')
    links += _tree_get_links(soup, 'a', 'href', '')
    links += _tree_get_links(soup, 'meta', 'content', '/')
    links += _tree_get_links(soup, 'script', 'src', '')
    links = [link for link in links if link.startswith("http")]

    _read_alexa()
    ranks = [_get_rank(link) for link in links]

    return {
        'mean': sum(ranks) / len(ranks),
        'sd': statistics.stdev(ranks)
    }


@register_feature(FeatureType.URL_WEBSITE, 'link_tree_features')
def link_tree(url: URLData):
    """
    Split the links into 30 sets as follows:
    Splits all links in the page by the following HTML tags:

        #. <a>
        #. <link>
        #. <script>
        #. <video>
        #. <img>
        #. <meta>

    Then divide into the following categories:

        #. social network links (Facebook, YouTube, Google, Twitter, Instagram, Pinterest)
        #. other https links,
        #. other http links,
        #. contains current domain,
        #. internal links.
        #. any URL

    Returns the size, mean length, and standard deviation of length of each set rounded to two decimal places

    Reference
    ----------

    Zhou, Xin, and Rakesh Verma. "Phishing sites detection from a web developer’s perspective using machine learning."
    Proceedings of the 53rd Hawaii International Conference on System Sciences. 2020.
    """
    # pylint: disable=too-many-locals
    domain = _extract_domain(url.final_url)
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')

    # get links from content
    link_link = _tree_get_links(soup, 'link', 'href', '')
    img_link = _tree_get_links(soup, 'img', 'src', '')
    video_link = _tree_get_links(soup, 'video', 'src', '')
    a_link = _tree_get_links(soup, 'a', 'src', '') + _tree_get_links(soup, 'a', 'href', '')
    meta_link = _tree_get_links(soup, 'meta', 'content', '/')
    script_link = _tree_get_links(soup, 'script', 'src', '')

    # extract features: size, mean, standard deviation
    link_features = _extract_tree_features(link_link, domain)
    img_features = _extract_tree_features(img_link, domain)
    video_features = _extract_tree_features(video_link, domain)
    a_features = _extract_tree_features(a_link, domain)
    meta_features = _extract_tree_features(meta_link, domain)
    script_features = _extract_tree_features(script_link, domain)

    features = {}
    _add_features(features, link_features, 'link')
    _add_features(features, img_features, 'img')
    _add_features(features, video_features, 'video')
    _add_features(features, a_features, 'a')
    _add_features(features, meta_features, 'meta')
    _add_features(features, script_features, 'script')
    return features


_ALEXA_DATA = None


def _read_alexa():
    # pylint: disable=global-statement
    global _ALEXA_DATA
    if _ALEXA_DATA is None:
        file_folder = pathlib.Path(__file__).parent.absolute()
        path = os.path.join(file_folder, 'alexa-top-1m.csv')
        with open(path) as f:
            reader = csv.DictReader(f, fieldnames=['rank', 'domain'])
            _ALEXA_DATA = {row["domain"]: row['rank'] for row in reader}


def _extract_domain(url):
    tld_extract = tldextract.extract(url)
    return f'{tld_extract.domain}.{tld_extract.suffix}'


def _tree_get_links(soup, tag, source, identifier):
    return [link.get(source) for
            link in soup.findAll(tag) if
            link.get(source) is not None and
            identifier in link.get(source)]


def _get_rank(url):
    # thresholds = [1000, 10000, 100000, 500000, 1000000, 5000000]
    # Alexa data only goes up to 1000000
    thresholds = [1000, 10000, 100000, 500000]
    domain = _extract_domain(url)
    if domain in _ALEXA_DATA:
        alexa_rank = int(_ALEXA_DATA[domain])
        for i, threshold in enumerate(thresholds, 1):
            if alexa_rank < threshold:
                return i
        return 7
    return 8


SOCIAL_DOMAINS = ['google.com', 'facebook.com', 'twitter.com', 'pinterest.com', 'instagram.com']


# Get size, mean, SD for a set of links
def _extract_tree_features(links, domain):
    features = []
    type_1 = type_2 = type_3 = type_4 = type_5 = []
    for link in links:
        link_domain = _extract_domain(link)
        if domain in link:
            type_1.append(link)
        elif link_domain in SOCIAL_DOMAINS:
            type_2.append(link)
        elif link.startswith("https:"):
            type_3.append(link)
        elif link.startswith("http:"):
            type_4.append(link)
        else:
            type_5.append(link)
    features.append(_extract_by_type(type_1))
    features.append(_extract_by_type(type_2))
    features.append(_extract_by_type(type_3))
    features.append(_extract_by_type(type_4))
    features.append(_extract_by_type(type_5))
    return features


def _extract_by_type(link_list):
    block_size = len(link_list)
    block_mean = 0
    block_std = 0
    links_length = [len(link) for link in link_list]
    if len(links_length) > 0:
        block_mean = round(statistics.mean(links_length), 2)
    if len(links_length) > 1:
        block_std = round(statistics.stdev(links_length), 2)
    return [block_size, block_mean, block_std]


def _add_features(feature_dict: dict, features, tag):
    # features = [[3],[3],[3],[3],[3]], ..., [[],[],[],[],[]]
    # T1 - T5: 5 * 3
    name_list = ['size', 'mean', 'SD']
    for i, set_f in enumerate(features):
        # size mean, SD
        for name, value in zip(name_list, set_f):
            feature_name = f"{tag}_T{i}_{name}"
            feature_dict[feature_name] = value
