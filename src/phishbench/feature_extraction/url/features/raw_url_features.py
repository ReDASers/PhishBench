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
    """
    The length of the domain of the url
    """
    return len(url.parsed_url.hostname)


@register_feature(FeatureType.URL_RAW, 'number_of_digits')
def num_digits(url: URLData):
    """
    The number of digits in the url
    """
    return sum(c.isdigit() for c in url.raw_url)


@register_feature(FeatureType.URL_RAW, 'number_of_dots')
def num_dots(url: URLData):
    """
    The number of times the `.` character occurs in the url
    """
    return url.raw_url.count('.')


@register_feature(FeatureType.URL_RAW, 'url_length')
def url_length(url: URLData):
    """
    The length of the url.
    """
    return len(url.raw_url)


@register_feature(FeatureType.URL_RAW, 'special_pattern')
def special_pattern(url: URLData):
    """
    Whether or not the string `?gws_rd=ssl` appears in the url
    """
    return "?gws_rd=ssl" in url.raw_url


@register_feature(FeatureType.URL_RAW, 'top_level_domain')
def top_level_domain(url: URLData):
    """
    The top level domain of the url
    """
    tld = tldextract.extract(url.raw_url).suffix
    return tld


@register_feature(FeatureType.URL_RAW, 'is_common_tld')
def is_common_tld(url: URLData):
    """
    Whether or not the tld is one of: .com, .net, .org, .edu, .mil, .gov, .co, .biz", .info, .me
    """
    common_tld_list = ["com", "net", "org", "edu", "mil", "gov", "co", "biz", "info", "me"]
    tld = tldextract.extract(url.raw_url).suffix
    return tld in common_tld_list


@register_feature(FeatureType.URL_RAW, 'is_ip_addr')
def is_ip_addr(url: URLData):
    """
    Whether or not the url is an IPv4 address
    """
    match = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", url.parsed_url.hostname)
    return match is not None


@register_feature(FeatureType.URL_RAW, 'has_https')
def has_https(url: URLData):
    """
    Whether or not the url is https
    """
    return url.parsed_url.scheme.startswith('https')


@register_feature(FeatureType.URL_RAW, 'has_at_symbol')
def has_at_symbol(url: URLData):
    """
    Whether or not the character `@` is in the url
    """
    return '@' in url.raw_url


@register_feature(FeatureType.URL_RAW, 'null_in_domain')
def null_in_domain(url: URLData):
    """
    Whether or not the string `null` is in the url (ignoring case)
    """
    return 'null' in url.parsed_url.hostname.lower()


@register_feature(FeatureType.URL_RAW, 'number_of_dashes')
def number_of_dashes(url: URLData):
    """
    The number of dashes in the url
    """
    return url.raw_url.count('-')


@register_feature(FeatureType.URL_RAW, 'number_of_slashes')
def number_of_slashes(url: URLData):
    """
    The number forward or backward slashes in the url
    """
    return url.raw_url.count('/') + url.raw_url.count('\\')


@register_feature(FeatureType.URL_RAW, 'http_in_middle')
def http_in_middle(url: URLData):
    """
    Whether or not the string 'http' occurs in the middle of the url.
    """
    match = re.match(".+http.+", url.raw_url)
    return match is not None


@register_feature(FeatureType.URL_RAW, 'has_port')
def has_port(url: URLData):
    """
    Whether or not the URL has a port number
    """
    return url.parsed_url.port is not None


@register_feature(FeatureType.URL_RAW, 'num_punctuation')
def num_punctuation(url: URLData):
    """
    Number of punctuation. Punctuation is defined as `string.punctuation`
    """
    return sum(x in string.punctuation for x in url.raw_url)


@register_feature(FeatureType.URL_RAW, 'digit_letter_ratio')
def digit_letter_ratio(url: URLData):
    """
    Number of digits divided by number of letters
    """
    url_num_digits = sum(c.isdigit() for c in url.raw_url)
    url_num_letters = sum(c.isalpha() for c in url.raw_url)
    return url_num_digits / url_num_letters

# region Character Distribution


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
        counts[int(ord(x) - ord('a'))] += 1
    num_letters = len(text)
    counts = [x/num_letters for x in counts]
    return counts


@register_feature(FeatureType.URL_RAW, 'char_dist')
def char_dist(url: URLData):
    """
    The character distribution of the url
    """
    url_char_distance = _calc_char_dist(url.raw_url)
    return {"char_distribution_{}".format(character): value for
            value, character in zip(url_char_distance, string.ascii_lowercase)}


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


# endregion

@register_feature(FeatureType.URL_RAW, 'consecutive_numbers')
def consecutive_numbers(url: URLData):
    """
    The sum of squares of the length of substrings that are consecutive numbers
    """
    matches = re.findall(r'\d+', url.raw_url)
    return sum((len(x)**2 for x in matches))
