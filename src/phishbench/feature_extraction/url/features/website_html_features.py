"""
This module contains features related to the html of a website
"""
from bs4 import BeautifulSoup
from tldextract import tldextract

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


@register_feature(FeatureType.URL_WEBSITE, 'number_object_tags')
def number_object_tags(url: URLData):
    """
    The number of object tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('object'))


@register_feature(FeatureType.URL_WEBSITE, 'number_of_video')
def number_of_video(url: URLData):
    """
    The number of video tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('video'))


@register_feature(FeatureType.URL_WEBSITE, 'number_of_audio')
def number_of_audio(url: URLData):
    """
    The number of audio tags
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    return len(soup.find_all('audio'))


@register_feature(FeatureType.URL_WEBSITE, 'number_of_hidden_object')
def number_of_hidden_object(url: URLData):
    """
    The number of objects of height or width 0
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    object_tags = soup.find_all('object')
    count = 0
    for tag in object_tags:
        if tag.get('height') == 0 or tag.get('width') == 0:
            count += 1
    return count


@register_feature(FeatureType.URL_WEBSITE, 'number_of_hidden_div')
def number_of_hidden_div(url: URLData):
    """
    The number of div of height or width 0
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    object_tags = soup.find_all('div')
    count = 0
    for tag in object_tags:
        if tag.get('height') == 0 or tag.get('width') == 0:
            count += 1
    return count


@register_feature(FeatureType.URL_WEBSITE, 'number_of_hidden_input')
def number_of_hidden_input(url: URLData):
    """
    The number of hidden input fields
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    object_tags = soup.find_all('input')
    hidden = [1 for tag in object_tags if tag.get('type') == "hidden"]
    return len(hidden)


@register_feature(FeatureType.URL_WEBSITE, 'number_of_hidden_iframe')
def number_of_hidden_iframe(url: URLData):
    """
    The number of iframe of height or width 0
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    object_tags = soup.find_all('iframe')
    count = 0
    for tag in object_tags:
        if tag.get('height') == 0 or tag.get('width') == 0:
            count += 1
    return count


@register_feature(FeatureType.URL_WEBSITE, 'number_of_hidden_svg')
def number_of_hidden_svg(url: URLData):
    """
    The number of iframe of height or width 0
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    object_tags = soup.find_all('svg')
    hidden = [1 for tag in object_tags if tag.get('aria-hidden') == "true"]
    return len(hidden)


_CONTENT_LIST = ['audio', 'embed', 'iframe', 'img', 'input', 'script', 'source', 'track', 'video']


@register_feature(FeatureType.URL_WEBSITE, 'number_of_external_content')
def number_of_external_content(url: URLData):
    """
    The number of external contents
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    url_extracted = tldextract.extract(url.final_url)
    local_domain = f'{url_extracted.domain}.{url_extracted.suffix}'
    outbound_count = 0

    tags = soup.find_all(_CONTENT_LIST)
    for tag in tags:
        src_address = tag.get('src')
        if src_address is not None:
            if src_address.startswith("//"):
                src_address = "http:" + src_address
            if src_address.lower().startswith(("https", "http")):
                extracted = tldextract.extract(src_address)
                link_domain = '{}.{}'.format(extracted.domain, extracted.suffix)
                if link_domain != local_domain:
                    outbound_count += 1
    return outbound_count


@register_feature(FeatureType.URL_WEBSITE, 'number_of_internal_content')
def number_of_internal_content(url: URLData):
    """
    The number of internal contents
    """
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    url_extracted = tldextract.extract(url.final_url)
    local_domain = f'{url_extracted.domain}.{url_extracted.suffix}'
    inbound_count = 0

    tags = soup.find_all(_CONTENT_LIST)
    for tag in tags:
        src_address = tag.get('src')
        if src_address is not None:
            if src_address.startswith("//"):
                src_address = "http:" + src_address
            if src_address.lower().startswith(("https", "http")):
                extracted = tldextract.extract(src_address)
                filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                if filtered_link == local_domain:
                    inbound_count = inbound_count + 1
            else:
                inbound_count = inbound_count + 1
    return inbound_count


@register_feature(FeatureType.URL_WEBSITE, 'number_of_internal_links')
def number_of_internal_links(url: URLData):
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')
    url_extracted = tldextract.extract(url.final_url)
    local_domain = '{}.{}'.format(url_extracted.domain, url_extracted.suffix)
    inbound_href_count = 0

    tags = soup.find_all(['a', 'area', 'base', 'link'])
    links = [tag.get('href') for tag in tags if tag.get('href') is not None]
    for link in links:
        if link.startswith("//"):
            link = "http:" + link
        if link.lower().startswith(("https", "http")):
            extracted = tldextract.extract(link)
            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
            if filtered_link == local_domain:
                inbound_href_count = inbound_href_count + 1
        else:
            inbound_href_count = inbound_href_count + 1
    return inbound_href_count

