import re

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
