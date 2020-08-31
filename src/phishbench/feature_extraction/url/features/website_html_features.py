from ...reflection import FeatureType, register_feature
from ....input import URLData

from bs4 import BeautifulSoup


@register_feature(FeatureType.URL_WEBSITE, 'number_of_tags')
def number_of_tags(url: URLData):
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all())