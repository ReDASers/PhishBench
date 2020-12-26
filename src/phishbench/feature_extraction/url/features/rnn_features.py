"""
Features from Visualizing and Interpreting RNN Models 
in URL-based Phishing Detection

https://dl.acm.org/doi/pdf/10.1145/3381991.3395602
"""

from phishbench.feature_extraction.reflection import * 
from phishbench.input import URLData


@register_feature(FeatureType.URL_RAW, 'HISC_whole')
def HISC_whole(url: URLData):
    """
    HISC_whole characters count in an entire URL
    HISC_whole = {{@xQ+]M&=<}#[?|‘} 
    """
    HISC_WHOLE = r'{{@xQ+]M&=<}#[?|\''
    return sum([x in HISC_WHOLE for x in url.raw_url])


@register_feature(FeatureType.URL_RAW, 'HISC_host')
def HISC_host(url: URLData):
    """
    HISC_host characters count in the host part of the URL
    
    HISC_host = {XznGR%rmNM=DIZc:}
    """
    HISC_HOST = r'XznGR%rmNM=DIZc:'
    return sum([x in HISC_HOST for x in url.parsed_url.hostname])


@register_feature(FeatureType.URL_RAW, 'HISC_path')
def HISC_path(url: URLData):
    HISC_PATH = r'Y{x+]p!=}#[|:h'
    return sum([x in HISC_PATH for x in url.parsed_url.path])


@register_feature(FeatureType.URL_RAW, 'HISC_qs')
def HISC_qs(url: URLData):
    HISC_QS = r'5)-x+]M=}D#[?|\'(h˜}'
    return sum([x in HISC_QS for x in url.parsed_url.query])
