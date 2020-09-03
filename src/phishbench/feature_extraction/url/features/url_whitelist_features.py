"""
This module contains whitelist features
"""
import tldextract

from ...reflection import FeatureType, register_feature
from ....input import URLData

BRANDS = ['microsoft', 'paypal', 'netflix', 'bankofamerica', 'wellsfargo', 'facebook', 'chase', 'orange', 'dhl',
          'dropbox', 'docusign', 'adobe', 'linkedin', 'apple', 'google', 'banquepopulaire', 'alibaba',
          'comcast', 'credit', 'agricole', 'yahoo', 'at', 'nbc', 'usaa', 'americanexpress', 'cibc', 'amazon',
          'ing', 'bt']


@register_feature(FeatureType.URL_RAW, 'brand_in_url')
def brand_in_url(url: URLData):
    """
    Whether or not the name of popular phishing targets are in the URL
    """
    return any(brand in url.raw_url for brand in BRANDS)


@register_feature(FeatureType.URL_RAW, 'is_whitelisted')
def is_whitelisted(url: URLData):
    """
    Whether or not the domain is one of the targeted brands
    """
    domain = tldextract.extract(url.raw_url).domain
    return domain in BRANDS
