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
