
from ...reflection import FeatureType, register_feature
from ....input import URLData


@register_feature(FeatureType.URL_WEBSITE, 'is_redirect')
def is_redirect(url: URLData):
    return url.raw_url.strip() != url.final_url.strip()
