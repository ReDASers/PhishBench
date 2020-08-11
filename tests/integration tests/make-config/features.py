from phishbench.feature_extraction.reflection import register_feature, FeatureType


@register_feature(FeatureType.EMAIL_HEADER, "test_feature")
def test(header):
    return 1
