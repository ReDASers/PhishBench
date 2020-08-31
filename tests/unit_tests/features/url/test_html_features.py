import unittest
from unittest.mock import patch

from phishbench import Features
from phishbench.feature_extraction.reflection import FeatureType
from phishbench.feature_extraction.url.features import html_features
from tests import mock_objects


class TestHTMLReflectionFeatures(unittest.TestCase):

    def test_is_redirect_true(self):
        # https://bit.ly/2Ef4uAS captured on 8/30/2020
        # redirects to https://en.wikipedia.org/wiki/Eastman_Kodak_Co._v._Image_Technical_Services,_Inc.
        test_url = mock_objects.get_mock_urldata('wikipedia_redirect')
        result = website_features.is_redirect(test_url)
        self.assertTrue(result)

    def test_is_redirect_false(self):
        # https://microsoft.com captured on 8/30/2020
        test_url = mock_objects.get_mock_urldata('microsoft')
        result = website_features.is_redirect(test_url)
        self.assertFalse(result)


@patch('phishbench.utils.phishbench_globals.config', new_callable=mock_objects.get_mock_config)
class TestHTMLFeatures(unittest.TestCase):

    def test_HTML_number_of_tags(self, mock_config):
        mock_config[FeatureType.URL_WEBSITE.value]["number_of_tags"] = "True"
        list_features = {}
        list_time = {}

        Features.HTML_number_of_tags(None, list_features, list_time)

        self.assertEqual(list_features["number_of_tags"], 0, 'incorrect number_of_tags')

    def test_HTML_number_of_hidden_svg(self, mock_config):
        mock_config[FeatureType.URL_WEBSITE.value]["number_of_hidden_svg"] = "True"
        list_features = {}
        list_time = {}

        soup = mock_objects.get_soup('test_1.html')

        Features.HTML_number_of_hidden_svg(soup, list_features, list_time)

        self.assertEqual(list_features["number_of_hidden_svg"], 1, 'incorrect number_of_svgs')

    def test_HTML_number_of_hidden_input(self, mock_config):
        mock_config[FeatureType.URL_WEBSITE.value]["number_of_hidden_input"] = "True"
        list_features = {}
        list_time = {}

        soup = mock_objects.get_soup('test_1.html')

        Features.HTML_number_of_hidden_input(soup, list_features, list_time)

        self.assertEqual(list_features["number_of_hidden_input"], 1, 'incorrect number_of_inputs')

