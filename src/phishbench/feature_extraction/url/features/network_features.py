"""
This module contains the built-in network features
"""
from ...reflection import FeatureType, register_feature
from ....input import URLData


@register_feature(FeatureType.URL_NETWORK, 'creation_date')
def creation_date(url: URLData):
    creation = url.domain_whois['creation_date']
    print(creation)
    if isinstance(creation, list):
        return creation[0].timestamp()
    return -1


@register_feature(FeatureType.URL_NETWORK, 'as_number')
def as_number(url: URLData):
    ip_whois = url.ip_whois[0]
    if 'asn' in ip_whois:
        return int(ip_whois['asn'])
    return -1