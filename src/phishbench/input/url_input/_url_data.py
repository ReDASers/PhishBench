"""
This is an internal module containing the implementation of the `URLData` model
"""
import re
import time
import urllib.error
import urllib.request
from typing import Optional, List, Dict
from urllib.parse import urlparse, ParseResult

import dns.resolver
import requests
import whois  # python-whois
from dns.exception import DNSException
from ipwhois import IPWhois
from ipwhois.exceptions import BaseIpwhoisException
from requests import HTTPError
from tldextract import tldextract

DNS_QUERY_TYPES = [
    'NONE',
    'A',
    'NS',
    'CNAME',
    'PTR',
    'MX',
    'SRV',
    'IXFR',
    'AXFR',
    'HINFO',
    'TLSA',
    'URI'
]
IPV4_REGEX = re.compile(r'^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.)'
                        r'{3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$')


def is_ip_address(url):
    """
    Checks whether or not a URL is an IPv4 address
    Parameters
    ----------
    url: str
        The url to check
    Returns
    -------
    result: bool
        Whether or not `str` is an IPv4 address
    """
    return bool(IPV4_REGEX.match(url))


# pylint: disable=too-many-instance-attributes
class URLData:
    """
    A data model representing a URL and its associated website.

    Attributes
    ----------
    raw_url: str
        The raw URL represented by this object
    parsed_url: ParseResult
        The parsed URL
    downloaded_website: str or None
        The downloaded HTML of the website pointed to by this URL. `None` if the website has not been downloaded.
    dns_results: Dict[str, List[str]] or None
        The results of the DNS queries as a dictionary. The key is the query type, and the values are a list of
        responses. `None` if the DNS information has not been looked up.
    ip_whois: List[Dict] or None
        The WHOIS results for each ip associated with this URL. `None` if the WHOIS results have not been looked up.
    domain_whois: whois.parser.WhoisEntry or None
        The WHOIS results for the domain. `None` if the WHOIS results have not been looked up.
    final_url: str, or None
        The URL of the website downloaded after all redirects have been processed.
    load_time: float
        The time in seconds it took to load the website. `-1` if the website was not downloaded.
    website_headers: http.client.HTTPMessage or None
        The headers received when downloading the website. `None` if the website was not downloaded.
    """

    def __init__(self, url: str, download_url=True):
        """

        Parameters
        ----------
        url: str
            The URL represented by this class.
        download_url: bool, optional
            Whether or not to download the website `url` points to, along with the DNS and whois info.
        """
        self.raw_url: str = url.strip()
        if not self.raw_url:
            raise ValueError("URL cannot be empty")
        self.parsed_url: ParseResult = urlparse(url)
        if not self.parsed_url.scheme:
            # If urlparse doesn't see // at the start of the hostname, it assumes that the url is relative
            self.parsed_url = urlparse('//' + url)
        self.downloaded_website: Optional[str] = None
        self.dns_results: Optional[Dict[str, List[str]]] = None
        self.ip_whois = None
        self.domain_whois = None
        self.final_url = None
        self.load_time = -1
        self.website_headers = None
        if download_url:
            self.download_website()
            self.lookup_dns()
            self.lookup_whois()

    def lookup_dns(self, nameservers: Optional[List[str]] = None):
        """
        Looks up and stores the DNS information for this URL.

        Parameters
        ----------
        nameservers: Optional[List[str]], optional
            The nameservers to use when looking up the DNS info. If nameservers are not provided, then PhishBench
            will use the system default nameservers.
        """
        if self.downloaded_website:
            final_parsed = urlparse(self.final_url)
            lookup_url = final_parsed.hostname
        else:
            lookup_url = self.parsed_url.hostname
        resolver = dns.resolver.get_default_resolver()
        if nameservers:
            resolver.nameservers = nameservers
        resolver.timeout = 1
        resolver.lifetime = 3

        self.dns_results = {}
        for query_type in DNS_QUERY_TYPES:
            try:
                answers = resolver.query(lookup_url, query_type)
                responses = [a.to_text() for a in answers]
                self.dns_results[query_type] = responses
            except DNSException:
                pass

    def lookup_whois(self, nameservers: Optional[List[str]] = None):
        """
        Looks up and stores the whois information for this URL.
        Parameters
        ----------
        nameservers: List[str] or None, optional
            The nameservers to use when looking up the DNS info. If name servers are not provided, then PhishBench will
            use the system default nameservers.
        """
        domain = ".".join(tldextract.extract(self.raw_url)[-2:])
        self.ip_whois = []
        if is_ip_address(domain):
            whois_client = IPWhois(domain)
            whois_result = whois_client.lookup_whois(get_referral=True)
            self.ip_whois.append(whois_result)
            return

        if not self.dns_results:
            self.lookup_dns(nameservers)

        if "A" in self.dns_results:
            ips = self.dns_results['A']
            for ip_address in ips:
                whois_client = IPWhois(ip_address)
                try:
                    whois_result = whois_client.lookup_whois(asn_methods=['dns', 'whois', 'http'], get_referral=True)
                    self.ip_whois.append(whois_result)
                except BaseIpwhoisException:
                    pass
        try:
            self.domain_whois = whois.whois(domain)
        except ConnectionError:
            self.domain_whois = whois.whois(domain, command=True)

    def download_website(self):
        """
        Download the website pointed to by this `URLData`
        """
        response = requests.head(self.raw_url, headers=_setup_request_headers(), timeout=20)
        if response.status_code >= 400:
            raise HTTPError("Status code not OK!")
        response.close()
        start_time = time.time()
        website = urllib.request.urlopen(self.raw_url)
        self.load_time = time.time() - start_time

        self.final_url = website.geturl()

        content: bytes = website.read()
        self.website_headers = website.info()
        content_type = self.website_headers['Content-Type']
        if ';' in content_type:
            content_type, encoding = content_type.split(';')
        else:
            encoding = 'encoding=utf-8'
        if content_type.startswith('text'):
            encoding = encoding.split('=')[1].strip()
            self.downloaded_website = content.decode(encoding)

    def __str__(self):
        return self.raw_url


def _setup_request_headers():
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '
                             'AppleWebKit/537.11 (KHTML, like Gecko) '
                             'Chrome/23.0.1271.64 Safari/537.11',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
               'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
               'Accept-Encoding': 'none',
               'Accept-Language': 'en-US,en;q=0.8',
               'Connection': 'keep-alive'}
    return headers
