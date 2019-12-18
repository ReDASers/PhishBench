from urllib.parse import urlparse
import os
import time
from urllib.request import urlopen
import urllib
import whois
from Cython.Tempita._tempita import html
from ipwhois import IPWhois
from pprint import pprint
import socket
import dns.resolver
import requests
import pickle
import datetime
from bs4 import BeautifulSoup
import sys
import re
import logging
import traceback
import tldextract
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import DesiredCapabilities
import configparser

logger = logging.getLogger('root')
whois_info = {}

config=configparser.ConfigParser()
config.read('Config_file.ini')

class HTTPResponse:
    def __init__(self):
        self.headers = {}
        self.html = ""
        self.url = ""

def is_IP_address(domain):
    if re.match("^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) == None:
        return False
    else:
        return True

def dns_lookup(domain):
    ids = ['NONE',
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
    lists=[]
    resolver = dns.resolver.Resolver()
    resolver.timeout = 3
    resolver.lifetime = 3
    for a in ids:
        try:
            answers = resolver.query(domain, a)
            for rdata in answers:
                val=a + ' : '+ rdata.to_text()
                lists.append(val)

        except Exception as e:
            pass
    return lists

def extract_dns_info(url, list_time):
    dns_lookup_output = ''
    if config["Network_Features"]["network_features"] == "True":
        try:
            extracted = tldextract.extract(url)
            parsed_url = urlparse(url)
            complete_domain = '{uri.hostname}'.format(uri=parsed_url)
        except Exception as e:
            logger.warning("Exception: Domain Error: {}".format(e))
            domain=''
            complete_domain=''

        if complete_domain:
            t0 = time.time()
            try:
                dns_lookup_output=dns_lookup(complete_domain)
                list_time['dns_lookup_time'] = time.time() - t0
            except Exception as e:
                dns_lookup_output=''
        else:
            dns_lookup_output=''

    return dns_lookup_output

def extract_whois(url, list_time): 
    IPs = ''
    ipwhois = None 
    whois_output = None
    domain = ''
    try:
        extracted = tldextract.extract(url)
        parsed_url = urlparse(url)
        complete_domain = '{uri.hostname}'.format(uri=parsed_url)
        domain = "{}.{}".format(extracted.domain, extracted.suffix)
    except Exception as e:
        logger.warning("Exception: Domain Error: {}".format(e))
        domain=''
        complete_domain=''

    if config["Network_Features"]["network_features"] == "True" and (config["Network_Features"]["creation_date"] == "True" or config["Network_Features"]["expiration_date"] == "True" or config["Network_Features"]["updated_date"] == "True"):
        if complete_domain:
            try:
                try:
                    IPs = list(map(lambda x: x[4][0], socket.getaddrinfo(complete_domain, 80, type=socket.SOCK_STREAM)))
                except socket.gaierror:
                    IPs = list(map(lambda x: x[4][0], socket.getaddrinfo("www." + complete_domain, 80, type=socket.SOCK_STREAM)))

                t0 = time.time()
                for ip in IPs:
                    obj = IPWhois(ip)
                    ipwhois = obj.lookup_whois(get_referral=True)
                list_time['ipwhois_time'] = time.time() - t0
            except Exception as e:
                logger.warning("Exception: ipwhois Error: {}".format(e))
                IPs=''
                ipwhois=''

            if not is_IP_address(complete_domain) and domain:
                t0 = time.time()
                try:
                    if domain in whois_info:
                        whois_output = whois_info[domain]
                    else:
                        whois_output = whois.whois(domain)
                        whois_info[domain] = whois_output
                    time.sleep(5)
                except Exception as e:
                    logger.warning("Exception whois: Domain {}. Error: {}".format(domain, e))
                    whois_output=''
                list_time['whois_time'] = time.time() - t0
            else:
                whois_output=''
        else:
            IPs=''
            ipwhois=''
            whois_output=''

    return IPs, ipwhois, whois_output, domain

def download_url(rawurl, list_time):
    html = ''
    Error = 0
    http_response = HTTPResponse()
    if config["HTML_Features"]["HTML_features"] == "True": 
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
        chrome_options = Options()
        chrome_options.set_headless() 
        desired_capabilities = DesiredCapabilities.CHROME.copy()
        desired_capabilities['loggingPrefs'] = { 'browser':'ALL' }

        url = rawurl.strip().rstrip('\n')
        if url == '':
            Error=1
            logger.warning("Empty URL")
            return http_response, http_response.html, Error
        try:
            t0 = time.time()
            browser = webdriver.Chrome(executable_path=os.path.abspath('chromedriver'), chrome_options=chrome_options, desired_capabilities=desired_capabilities)
            browser.set_page_load_timeout(10)
            browser.get(url)
            log = browser.get_log('browser')
            http_response.html = browser.page_source
            http_response.url = browser.current_url
            browser.close()
            response = requests.head(url, headers = headers, timeout = 20)
            http_response.headers = response.headers
            response.close()
            if log:
                p = re.compile('.* status of ([0-9]+) .*')
                match = p.match(log[0]['message'])
                if match:
                    Error=1
                    logger.warning("HTTP response code is not OK: {}".format(match.groups()[0]))
                    return http_response, http_response.html, Error
            else:
                parsed = BeautifulSoup(http_response.html, 'html.parser')
                language = parsed.find("html").get('lang')
                if language != None and not language.startswith('en'):
                    Error=1
                    logger.warning("Website's language is not English")
                    return http_response, http_response.html, Error
        except Exception as e:
            logger.warning("Exception HTML: {}. Error :{}".format(url, e))
            logger.warning("html, content=''")
            logger.debug(traceback.format_exc())
            http_response.html = ''
            http_response.url = url

        list_time['html_time'] = time.time() - t0
    else:
        http_response.html = ''
        http_response.url = rawurl

    return http_response, http_response.html, Error

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True

def download_url_content(rawurl):
    html = urllib.request.urlopen('http://bgr.com/2014/10/15/google-android-5-0-lollipop-release/')
    soup = BeautifulSoup(html, "html5lib")
    data = soup.findAll(text=True)
    result = filter(visible, data)
    return list(result)

if __name__ == "__main__":
    data = download_url_content("http://docs.python-requests.org/en/master/user/quickstart/")
    logging.info(data)
