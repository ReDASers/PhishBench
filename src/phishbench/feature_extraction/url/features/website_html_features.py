from ...reflection import FeatureType, register_feature
from ....input import URLData

from bs4 import BeautifulSoup


@register_feature(FeatureType.URL_WEBSITE, 'number_of_tags')
def number_of_tags(url: URLData):
    """
    The number of tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all())


@register_feature(FeatureType.URL_WEBSITE, 'number_of_head')
def number_of_head(url: URLData):
    """
    The number of head tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('head'))


@register_feature(FeatureType.URL_WEBSITE, 'number_of_html')
def number_of_html(url: URLData):
    """
    The number of html tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('html'))


@register_feature(FeatureType.URL_WEBSITE, 'number_of_body')
def number_of_body(url: URLData):
    """
    The number of body tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('body'))


@register_feature(FeatureType.URL_WEBSITE, 'number_of_title')
def number_of_title(url: URLData):
    """
    The number of title tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('title'))
