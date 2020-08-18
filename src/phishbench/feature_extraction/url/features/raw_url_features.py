"""
This module contains the built-in raw url features
"""
import re
import string

import scipy.stats
import scipy.spatial
from tldextract import tldextract

from ...reflection import FeatureType, register_feature
from ....input import URLData


@register_feature(FeatureType.URL_RAW, 'domain_length')
def domain_length(url: URLData):
    return len(url.parsed_url.hostname)


@register_feature(FeatureType.URL_RAW, 'number_of_digits')
def num_digits(url: URLData):
    return sum(c.isdigit() for c in url.raw_url)


@register_feature(FeatureType.URL_RAW, 'number_of_dots')
def num_dots(url: URLData):
    return url.raw_url.count('.')


@register_feature(FeatureType.URL_RAW, 'url_length')
def url_length(url: URLData):
    return len(url.raw_url)


@register_feature(FeatureType.URL_RAW, 'special_pattern')
def special_pattern(url: URLData):
    return "?gws_rd=ssl" in url.raw_url


@register_feature(FeatureType.URL_RAW, 'is_common_tld')
def is_common_tld(url: URLData):
    common_tld_list = ["com", "net", "org", "edu", "mil", "gov", "co", "biz", "info", "me"]
    tld = tldextract.extract(url.raw_url).suffix
    return tld in common_tld_list


@register_feature(FeatureType.URL_RAW, 'is_ip_addr')
def is_ip_addr(url: URLData):
    print(url.parsed_url.hostname)
    match = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", url.parsed_url.hostname)
    return match is not None


@register_feature(FeatureType.URL_RAW, 'has_https')
def has_https(url: URLData):
    return url.parsed_url.scheme.startswith('https')


@register_feature(FeatureType.URL_RAW, 'has_at_symbol')
def has_at_symbol(url: URLData):
    return '@' in url.raw_url


@register_feature(FeatureType.URL_RAW, 'null_in_domain')
def null_in_domain(url: URLData):
    return 'null' in url.parsed_url.hostname.lower()


@register_feature(FeatureType.URL_RAW, 'number_of_dashes')
def number_of_dashes(url: URLData):
    return url.raw_url.count('-')


@register_feature(FeatureType.URL_RAW, 'http_in_middle')
def http_in_middle(url: URLData):
    match = re.match(".+http.+", url.raw_url)
    return match is not None


@register_feature(FeatureType.URL_RAW, 'has_port')
def has_port(url: URLData):
    return url.parsed_url.port is not None


@register_feature(FeatureType.URL_RAW, 'num_punctuation')
def num_punctuation(url: URLData):
    return sum(x in string.punctuation for x in url.raw_url)


_CHAR_DIST = [.08167, .01492, .02782, .04253, .12702, .02228, .02015, .06094, .06966, .00153, .00772, .04025, .02406,
              .06749, .07507, .01929, .00095, .05987, .06327, .09056, .02758, .00978, .02360, .00150, .01974, .00074]


def _calc_char_dist(text):
    """
    Computes the character distribution of the English letters in a string
    Parameters
    ----------
    str
        A string
    Returns
    -------
    A list containing the normalized character counts
    """
    text = re.sub(r'[^a-z]', '', text.lower())
    counts = [0] * 26
    for x in text:
        counts[int(x-'a')] += 1
    num_letters = len(text)
    counts = [x/num_letters for x in counts]
    return counts


@register_feature(FeatureType.URL_RAW, 'char_dist_kolmogorov_shmirnov')
def kolmogorov_shmirnov(url: URLData):
    """
    The Kolmogorov_Shmirnov statistic between the URL and the English character distribution
    """
    url_char_distance = _calc_char_dist(url.raw_url)
    return scipy.stats.ks_2samp(url_char_distance, _CHAR_DIST)[0]


@register_feature(FeatureType.URL_RAW, 'char_dist_kl_divergence')
def kullback_leibler(url: URLData):
    """
        The Kullback_Leibler divergence between the URL and the English character distribution
    """
    url_char_distance = _calc_char_dist(url.raw_url)
    return scipy.stats.entropy(url_char_distance, _CHAR_DIST)


@register_feature(FeatureType.URL_RAW, 'char_dist_euclidian_distance')
def euclidean_distance(url: URLData):
    """
        The Euclidean distance (L2 norm of u-v) between the URL and the English character distribution
    """
    url_char_distance = _calc_char_dist(url.raw_url)
    return scipy.spatial.distance.euclidean(url_char_distance, _CHAR_DIST)
