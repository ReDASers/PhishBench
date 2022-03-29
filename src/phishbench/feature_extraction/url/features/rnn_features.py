"""
Features from Visualizing and Interpreting RNN Models 
in URL-based Phishing Detection

https://dl.acm.org/doi/pdf/10.1145/3381991.3395602
"""

from phishbench.feature_extraction.reflection import * 
from phishbench.input import URLData


@register_feature(FeatureType.URL_RAW, 'hisc_whole')
def hisc_whole(url: URLData):
    """
    Number of characters from the set ``{@xQ+]M&=<}#[?|'`` in the URL

    Reference
    ----------

    Tao Feng and Chuan Yue. 2020. `Visualizing and Interpreting RNN Models in URL-based Phishing Detection.
    <https://doi.org/10.1145/3381991.3395602>`_
    """
    HISC_WHOLE = r'{{@xQ+]M&=<}#[?|\''
    return sum([x in HISC_WHOLE for x in url.raw_url])


@register_feature(FeatureType.URL_RAW, 'hisc_host')
def hisc_host(url: URLData):
    """
    Number of characters from the set ``XznGR%rmNM=DIZc:`` in the host section of the URL

    Reference
    ----------

    Tao Feng and Chuan Yue. 2020. `Visualizing and Interpreting RNN Models in URL-based Phishing Detection.
    <https://doi.org/10.1145/3381991.3395602>`_
    """
    if url.parsed_url.hostname is None:
        return 0
    HISC_HOST = r'XznGR%rmNM=DIZc:'
    return sum([x in HISC_HOST for x in url.parsed_url.hostname])


@register_feature(FeatureType.URL_RAW, 'hisc_path')
def hisc_path(url: URLData):
    """
    Number of characters from the set ``Y{x+]p!=}#[|:h`` in the path section of the URL

    Reference
    ----------

    Tao Feng and Chuan Yue. 2020. `Visualizing and Interpreting RNN Models in URL-based Phishing Detection.
    <https://doi.org/10.1145/3381991.3395602>`_
    """
    HISC_PATH = r'Y{x+]p!=}#[|:h'
    return sum([x in HISC_PATH for x in url.parsed_url.path])


@register_feature(FeatureType.URL_RAW, 'hisc_query')
def hisc_query(url: URLData):
    """
    Number of characters from the set ``5)-x+]M=}D#[?|\'(h˜}`` in the query section of the URL

    Reference
    ----------

    Tao Feng and Chuan Yue. 2020. `Visualizing and Interpreting RNN Models in URL-based Phishing Detection.
    <https://doi.org/10.1145/3381991.3395602>`_
    """
    HISC_QS = r'5)-x+]M=}D#[?|\'(h˜}'
    return sum([x in HISC_QS for x in url.parsed_url.query])
