"""
Tests for l-tree features
"""
import unittest

from phishbench.input import URLData
from phishbench.feature_extraction.url.features import link_tree_features
from tests import mock_objects

# pylint: disable=missing-function-docstring
# pylint: disable=no-value-for-parameter
# pylint: disable=too-many-public-methods

class TestLinkTreeFeatures(unittest.TestCase):

    def test_extract_domain(self):
        # pylint: disable=protected-access
        result = link_tree_features._extract_domain("abc.google.com/test")
        self.assertEqual('google.com', result)

    def test_ltree(self):
        url: URLData = mock_objects.get_mock_object("wikipedia_shortener_urldata")
        expected = mock_objects.get_mock_object('wikipedia-Ltree')

        result = link_tree_features.link_tree().extract(url)

        self.assertDictEqual(expected, result)

    def test_ranked_matrix(self):
        url: URLData = mock_objects.get_mock_object("reddit_urldata")

        result = link_tree_features.link_alexa_global_rank().extract(url)
        expected = {
            'mean': 5.4411764705882355,
            'sd': 3.178083992742128
        }
        self.assertDictEqual(expected, result)
