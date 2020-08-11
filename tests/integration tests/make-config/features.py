from phishbench.feature_extraction.email.reflection import register_feature, FeatureType


@register_feature(FeatureType.HEADER, "test_feature")
def test(header):
    return 1
