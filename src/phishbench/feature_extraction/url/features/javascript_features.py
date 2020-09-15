"""
This module contains built-in javascript features
"""
import re
from bs4 import BeautifulSoup

from ...reflection import register_feature, FeatureType
from ....input.url_input import URLData


@register_feature(FeatureType.URL_WEBSITE_JAVASCRIPT, 'number_of_exec')
def number_of_exec(url: URLData):
    """
    Number of `exec` calls in the embedded javascript
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    scripts = [script for script in soup.find_all('script')
               if script.get("type") is None or script.get("type") == 'text/javascript']
    counts = [len(re.findall(r'\Wexec\(', str(script))) for script in scripts]
    return sum(counts)


@register_feature(FeatureType.URL_WEBSITE_JAVASCRIPT, 'number_of_escape')
def number_of_escape(url: URLData):
    """
    Number of `escape` calls in the embedded javascript
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    scripts = [script for script in soup.find_all('script')
               if script.get("type") is None or script.get("type") == 'text/javascript']
    counts = [len(re.findall(r'\Wescape\(', str(script))) for script in scripts]
    return sum(counts)


@register_feature(FeatureType.URL_WEBSITE_JAVASCRIPT, 'number_of_eval')
def number_of_eval(url: URLData):
    """
    Number of `eval` calls in the embedded javascript
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    scripts = [script for script in soup.find_all('script')
               if script.get("type") is None or script.get("type") == 'text/javascript']
    counts = [len(re.findall(r'\Weval\(', str(script))) for script in scripts]
    return sum(counts)


@register_feature(FeatureType.URL_WEBSITE_JAVASCRIPT, 'number_of_link')
def number_of_link(url: URLData):
    """
    Number of `link` calls in the embedded javascript
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    scripts = [script for script in soup.find_all('script')
               if script.get("type") is None or script.get("type") == 'text/javascript']
    counts = [len(re.findall(r'\Wlink\(', str(script))) for script in scripts]
    return sum(counts)


@register_feature(FeatureType.URL_WEBSITE_JAVASCRIPT, 'number_of_unescape')
def number_of_unescape(url: URLData):
    """
    Number of `unescape` calls in the embedded javascript
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    scripts = [script for script in soup.find_all('script')
               if script.get("type") is None or script.get("type") == 'text/javascript']
    counts = [len(re.findall(r'\Wunescape\(', str(script))) for script in scripts]
    return sum(counts)


@register_feature(FeatureType.URL_WEBSITE_JAVASCRIPT, 'number_of_search')
def number_of_search(url: URLData):
    """
    Number of `search` calls in the embedded javascript
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    scripts = [script for script in soup.find_all('script')
               if script.get("type") is None or script.get("type") == 'text/javascript']
    counts = [len(re.findall(r'\Wsearch\(', str(script))) for script in scripts]
    return sum(counts)


@register_feature(FeatureType.URL_WEBSITE_JAVASCRIPT, 'number_of_set_timeout')
def number_of_set_timeout(url: URLData):
    """
    Number of `setTimeout` calls in the embedded javascript
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    scripts = [script for script in soup.find_all('script')
               if script.get("type") is None or script.get("type") == 'text/javascript']
    counts = [len(re.findall(r'\WsetTimeout\(', str(script))) for script in scripts]
    return sum(counts)


@register_feature(FeatureType.URL_WEBSITE_JAVASCRIPT, 'number_of_iframes_in_script')
def number_of_iframes_in_script(url: URLData):
    """
    Number of times the token `iframe` shows up in embedded javascript
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    scripts = [script for script in soup.find_all('script')
               if script.get("type") is None or script.get("type") == 'text/javascript']
    counts = [len(re.findall(r'iframe', str(script))) for script in scripts]
    return sum(counts)


@register_feature(FeatureType.URL_WEBSITE_JAVASCRIPT, 'number_of_event_attachment')
def number_of_event_attachment(url: URLData):
    """
    Whether or not the right click event has been modified.
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    scripts = [re.sub(r'\s', '', str(script)) for script in soup.find_all('script')]
    event_regex = re.compile(r'\.(?:addEventListener|attachEvent)\(')
    counts = [len(event_regex.findall(script)) for script in scripts]
    return sum(counts)


@register_feature(FeatureType.URL_WEBSITE_JAVASCRIPT, 'number_of_event_dispatch')
def number_of_event_dispatch(url: URLData):
    """
    Whether or not the right click event has been modified.
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    scripts = [re.sub(r'\s', '', str(script)) for script in soup.find_all('script')]
    event_regex = re.compile(r'\.(?:dispatchEvent|fireEvent)\(')
    counts = [len(event_regex.findall(script)) for script in scripts]
    return sum(counts)


@register_feature(FeatureType.URL_WEBSITE_JAVASCRIPT, 'right_click_modified')
def right_click_modified(url: URLData):
    """
    Whether or not the right click event has been modified.
    """
    text = re.sub(r'\s', '', url.downloaded_website)
    return "addEventListener('contextmenu'" in text
