"""
Contains features related to tokens in the url

References
PhishDef: URL Names Say It All
Detecting Malicious URLs Using Lexical Analysis
"""
import re

from ...reflection import FeatureType, register_feature
from ....input import URLData

_TOKEN_DELIMITER_REGEX = re.compile(r'[/\?\.=_&\-\']+')


@register_feature(FeatureType.URL_RAW, 'token_count')
def token_count(url: URLData):
    """
    The number of tokens in the path
    """
    tokens = _TOKEN_DELIMITER_REGEX.split(url.raw_url)
    return len(tokens)


@register_feature(FeatureType.URL_RAW, 'average_path_token_length')
def average_path_token_length(url: URLData):
    """
    Average length of tokens from the URL path
    """
    tokens = _TOKEN_DELIMITER_REGEX.split(url.parsed_url.path)
    if len(tokens[0]) == 0:
        tokens = tokens[1:]
    if len(tokens) == 0:
        return 0
    lengths = [len(token) for token in tokens]
    return sum(lengths) / len(tokens)


@register_feature(FeatureType.URL_RAW, 'average_domain_token_length')
def average_domain_token_length(url: URLData):
    """
    Average length of tokens from the URL domain
    """
    tokens = _TOKEN_DELIMITER_REGEX.split(url.parsed_url.hostname)
    lengths = [len(token) for token in tokens]
    return sum(lengths) / len(tokens)


@register_feature(FeatureType.URL_RAW, 'longest_domain_token_length')
def longest_domain_token_length(url: URLData):
    """
    Length of the length of tokens from the URL domain
    """
    tokens = _TOKEN_DELIMITER_REGEX.split(url.parsed_url.hostname)
    lengths = [len(token) for token in tokens]
    return max(lengths)
