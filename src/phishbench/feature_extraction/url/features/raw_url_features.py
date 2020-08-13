from ...reflection import FeatureType, register_feature
from ....input import URLData


@register_feature(FeatureType.URL_RAW, 'url_length')
def url_length(url: URLData):
    return len(url.raw_url)


@register_feature(FeatureType.URL_RAW, 'domain_length')
def domain_length(url: URLData):
    return len(url.parsed_url.hostname)
