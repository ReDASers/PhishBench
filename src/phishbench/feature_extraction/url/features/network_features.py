"""
This module contains the built-in network features
"""
from ...reflection import FeatureType, register_feature
from ....input import URLData


@register_feature(FeatureType.URL_NETWORK, 'creation_date')
def creation_date(url: URLData):
    creation = url.domain_whois['creation_date']
    if isinstance(creation, list):
        return creation[0].timestamp()
    return -1


@register_feature(FeatureType.URL_NETWORK, 'as_number')
def as_number(url: URLData):
    ip_whois = url.ip_whois[0]
    if 'asn' in ip_whois:
        return int(ip_whois['asn'])
    return -1


@register_feature(FeatureType.URL_NETWORK, 'number_name_server')
def number_name_server(url: URLData):
    if 'NS' in url.dns_results:
        return len(url.dns_results['NS'])
    return 0


@register_feature(FeatureType.URL_NETWORK, 'expiration_date')
def expiration_date(url: URLData):
    date = url.domain_whois['expiration_date']
    if isinstance(date, list):
        return date[0].timestamp()
    return -1


@register_feature(FeatureType.URL_NETWORK, 'updated_date')
def updated_date(url: URLData):
    date = url.domain_whois['updated_date']
    if isinstance(date, list):
        return date[0].timestamp()
    return -1
