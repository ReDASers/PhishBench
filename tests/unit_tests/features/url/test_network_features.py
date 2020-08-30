"""
Tests for network features
"""
import datetime
import unittest
from unittest.mock import patch, MagicMock

from phishbench.feature_extraction.url.features import network_features
from tests import mock_objects


# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

class TestNetworkFeatures(unittest.TestCase):

    def test_creation_date(self):
        expected = datetime.datetime(1991, 5, 2, 4, 0).timestamp()
        test_url = mock_objects.get_mock_object('microsoft_urldata')
        result = network_features.creation_date().extract(test_url)

        self.assertEqual(expected, result)

    def test_as_number(self):
        test_url = mock_objects.get_mock_object('microsoft_urldata')
        result = network_features.as_number().extract(test_url)

        self.assertEqual(16625, result)

    def test_number_name_server(self):
        test_url = mock_objects.get_mock_object('google_urldata')
        result = network_features.number_name_server().extract(test_url)

        self.assertEqual(4, result)

    def test_expiration_date(self):
        expected = datetime.datetime(2021, 5, 3, 4, 0).timestamp()
        test_url = mock_objects.get_mock_object('microsoft_urldata')
        result = network_features.expiration_date().extract(test_url)

        self.assertEqual(expected, result)

    def test_updated_date(self):
        expected = datetime.datetime(2020, 5, 20, 19, 54, 16).timestamp()
        test_url = mock_objects.get_mock_object('microsoft_urldata')
        result = network_features.updated_date().extract(test_url)

        self.assertEqual(expected, result)

    @patch('dns.resolver.query')
    def test_dns_ttl(self, dns_mock: MagicMock):
        test_url = mock_objects.get_mock_object('microsoft_urldata')
        dns_mock.return_value = mock_objects.get_mock_object('microsoft_dns_query')

        result = network_features.dns_ttl().extract(test_url)

        dns_mock.assert_called_once_with('www.microsoft.com', 'A')
        self.assertEqual(13, result)
