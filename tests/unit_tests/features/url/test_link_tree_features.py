"""
Tests for l-tree features
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

        result = link_tree_features.link_tree(url)

        self.assertDictEqual(expected, result)

    def test_ranked_matrix(self):
        url: URLData = mock_objects.get_mock_object("reddit_urldata")

        result = link_tree_features.ranked_matrix(url)
        expected = {
            'mean': 5.45925925925926,
            'sd': 3.182890273318344
        }
        self.assertDictEqual(expected, result)