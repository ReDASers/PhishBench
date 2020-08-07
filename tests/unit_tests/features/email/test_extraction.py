import unittest

from phishbench.feature_extraction.email.email_features import load_internal_features


class TestFeatureExtraction(unittest.TestCase):

    def test_load_internal_features(self):
        features = load_internal_features(filter_features=False)
        self.assertIsNotNone(features)
        self.assertGreater(len(features), 0)
