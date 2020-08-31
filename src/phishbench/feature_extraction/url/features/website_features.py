
from ...reflection import FeatureType, register_feature
from ....input import URLData


@register_feature(FeatureType.URL_WEBSITE, 'is_redirect')
def is_redirect(url: URLData):
    return url.raw_url.strip() != url.final_url.strip()




@register_feature(FeatureType.URL_WEBSITE, 'website_content_type')
def content_type_header(url: URLData):
    content_type = url.website_headers['Content-Type']
    if not content_type:
        return "text/html"
    if ';' in content_type:
        content_type, _ = content_type.split(';')
    return content_type


@register_feature(FeatureType.URL_WEBSITE, 'content_length')
def content_length_header(url: URLData):
    if'Content-Length' not in url.website_headers:
        return -1
    header_value = url.website_headers['Content-Length']
    return int(header_value)


@register_feature(FeatureType.URL_WEBSITE, 'x_powered_by')
def x_powered_by_header(url: URLData):
    if'X-Powered-By' not in url.website_headers:
        return ""
    return url.website_headers['X-Powered-By']
