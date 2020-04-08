import unittest
from unittest.mock import patch

from phishbench import Features
from tests.mock_objects import mock_objects

@patch('phishbench.utils.Globals.config', new_callable=mock_objects.get_mock_config)
class TestHTMLFeatures(unittest.TestCase):

    def test_HTML_number_of_tags(self, mock_config):
        mock_config["HTML_Features"]["number_of_tags"] = "True"
        list_features = {}
        list_time = {}

        Features.HTML_number_of_tags(None, list_features, list_time)

        self.assertEqual(list_features["number_of_tags"], 0, 'incorrect number_of_tags')