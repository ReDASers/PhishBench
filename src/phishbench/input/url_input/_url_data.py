import os.path
import pathlib
import platform
import re
import time
from collections import namedtuple
from urllib.parse import urlparse

import dns.resolver
import requests
import whois  # python-whois
from dns.exception import DNSException
from ipwhois import IPWhois
from ipwhois.exceptions import BaseIpwhoisException
from requests import HTTPError
from selenium import webdriver
from selenium.webdriver import DesiredCapabilities
from selenium.webdriver.chrome.options import Options

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
    return bool(IPV4_REGEX.match(url))


HTTPResponse = namedtuple('HTTPResponse', "headers html final_url log")


class URLData:

    def __init__(self, url: str, download_url=True):
        self.raw_url = url.strip()
        if not self.raw_url:
            raise ValueError("URL cannot be empty")
        if not url.startswith("http"):
            url = "http://" + url
        self.parsed_url = urlparse(url)

        self.downloaded_website = None
        self.dns_results = None
        self.ip_whois = None
        self.domain_whois = None
        if download_url:
            self.download_website()
            self.lookup_dns()
            self.lookup_whois()

    def lookup_dns(self, nameservers=None):
        if self.downloaded_website:
            final_parsed = urlparse(self.downloaded_website.final_url)
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

    def lookup_whois(self, nameservers=None):
        domain = self.parsed_url.hostname
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
        browser = _setup_browser()
        response = requests.head(self.raw_url, headers=_setup_request_headers(), timeout=20)
        if response.status_code >= 400:
            raise HTTPError("Status code not OK!")
        start_time = time.time()
        browser.get(self.raw_url)
        html_time = time.time() - start_time
        self.downloaded_website = HTTPResponse(
            log=browser.get_log('browser'),
            html=browser.page_source,
            final_url=browser.current_url,
            headers=response.headers
        )
        response.close()
        browser.close()
        return html_time

    def __str__(self):
        return self.raw_url


def _setup_browser():
    chrome_options = Options()
    chrome_options.headless = True
    chrome_options.add_argument('--log-level=3')
    desired_capabilities = DesiredCapabilities.CHROME.copy()
    desired_capabilities['loggingPrefs'] = {'browser': 'ALL'}
    chrome_path = pathlib.Path(__file__).parent.absolute()
    if platform.system() == 'Windows':
        chrome_path = os.path.join(chrome_path, 'chromedriver.exe')
    elif platform.system() == 'Linux':
        chrome_path = os.path.join(chrome_path, 'chromedriver_linux')
    else:
        chrome_path = os.path.join(chrome_path, 'chromedriver_mac')
    browser = webdriver.Chrome(executable_path=chrome_path, chrome_options=chrome_options,
                               desired_capabilities=desired_capabilities)
    browser.set_page_load_timeout(10)
    return browser


def _setup_request_headers():
    headers = requests.utils.default_headers()
    headers.update(
        {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:62.0) Gecko/20100101 Firefox/62.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q = 0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q = 0.7',
            'Keep-Alive': '300',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'Accept-Language': '*',
            'Accept-Encoding': 'gzip, deflate'
        }
    )
    return headers
