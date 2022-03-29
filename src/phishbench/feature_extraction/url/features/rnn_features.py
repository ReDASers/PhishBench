"""
Features from Visualizing and Interpreting RNN Models
in URL-based Phishing Detection

https://dl.acm.org/doi/pdf/10.1145/3381991.3395602
"""

from phishbench.feature_extraction.reflection import FeatureType, register_feature
from phishbench.input import URLData

HISC_WHOLE = r'{{@xQ+]M&=<}#[?|\''


@register_feature(FeatureType.URL_RAW, 'hisc_whole')
def hisc_whole(url: URLData):
    """
    Number of characters from the set ``{@xQ+]M&=<}#[?|'`` in the URL

    Reference
    ----------

    Tao Feng and Chuan Yue. 2020. `Visualizing and Interpreting RNN Models in URL-based Phishing Detection.
    <https://doi.org/10.1145/3381991.3395602>`_
    """
    return sum([x in HISC_WHOLE for x in url.raw_url])


HISC_HOST = r'XznGR%rmNM=DIZc:'


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
    return sum([x in HISC_HOST for x in url.parsed_url.hostname])


HISC_PATH = r'Y{x+]p!=}#[|:h'


@register_feature(FeatureType.URL_RAW, 'hisc_path')
def hisc_path(url: URLData):
    """
    Number of characters from the set ``Y{x+]p!=}#[|:h`` in the path section of the URL

    Reference
    ----------

    Tao Feng and Chuan Yue. 2020. `Visualizing and Interpreting RNN Models in URL-based Phishing Detection.
    <https://doi.org/10.1145/3381991.3395602>`
    """
    return sum([x in HISC_PATH for x in url.parsed_url.path])


HISC_QS = r'5)-x+]M=}D#[?|\'(h~}'


@register_feature(FeatureType.URL_RAW, 'hisc_query')
def hisc_query(url: URLData):
    """
    Number of characters from the set ``5)-x+]M=}D#[?|\'(h~}`` in the query section of the URL

    Reference
    ----------

    Tao Feng and Chuan Yue. 2020. `Visualizing and Interpreting RNN Models in URL-based Phishing Detection.
    <https://doi.org/10.1145/3381991.3395602>`_
    """
    return sum([x in HISC_QS for x in url.parsed_url.query])
