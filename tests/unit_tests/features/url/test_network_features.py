import datetime
import unittest

from phishbench.feature_extraction.url.features import network_features
from tests import mock_objects


class TestNetworkFeatures(unittest.TestCase):

    def test_creation_date(self):
        expected = datetime.datetime(1991, 5, 2, 4, 0).timestamp()
        test_url = mock_objects.get_mock_urldata('microsoft')
        result = network_features.creation_date(test_url)

        self.assertEqual(expected, result)

    def test_as_number(self):
        test_url = mock_objects.get_mock_urldata('microsoft')
        result = network_features.as_number(test_url)

        self.assertEqual(16625, result)

    def test_number_name_server(self):
        test_url = mock_objects.get_mock_urldata('google')
        result = network_features.number_name_server(test_url)

        self.assertEqual(4, result)
