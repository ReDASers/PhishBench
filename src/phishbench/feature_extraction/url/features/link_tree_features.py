from ...reflection import FeatureType, register_feature
from ....input import URLData


@register_feature(FeatureType.URL_WEBSITE, 'link_tree_features')
def link_tree_features(url: URLData):
    return {}
