from bs4 import BeautifulSoup

from ...reflection import FeatureType, register_feature
from ....input import URLData


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


@register_feature(FeatureType.URL_WEBSITE, 'number_of_iframe')
def number_of_iframe(url: URLData):
    """
    The number of iframe tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('iframe'))


@register_feature(FeatureType.URL_WEBSITE, 'number_of_input')
def number_of_input(url: URLData):
    """
    The number of input tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('input'))


@register_feature(FeatureType.URL_WEBSITE, 'number_of_img')
def number_of_img(url: URLData):
    """
    The number of img tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('img'))


@register_feature(FeatureType.URL_WEBSITE, 'number_of_scripts')
def number_of_script(url: URLData):
    """
    The number of script tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('script'))


@register_feature(FeatureType.URL_WEBSITE, 'number_of_anchor')
def number_of_anchor(url: URLData):
    """
    The number of anchor (<a>) tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('a'))


@register_feature(FeatureType.URL_WEBSITE, 'number_of_embed')
def number_of_embed(url: URLData):
    """
    The number of embed tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('embed'))
