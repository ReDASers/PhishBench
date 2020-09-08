"""
Tests for network features
"""
import unittest

from phishbench.input import URLData
from phishbench.feature_extraction.url.features import link_tree_features
from tests import mock_objects

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestLinkTreeFeatures(unittest.TestCase):

    def test_ltree(self):
        url: URLData = mock_objects.get_mock_object("wikipedia_shortener_urldata")
        expected = mock_objects.get_mock_object('wikipedia-Ltree')

        result = link_tree_features.link_tree_features(url)

        self.assertDictEqual(expected, result)
