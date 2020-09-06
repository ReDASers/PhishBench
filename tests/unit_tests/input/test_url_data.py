"""
This module tests the `phishbench.input.url_input.URLData` data module
"""
import unittest
from unittest.mock import patch, MagicMock
from http.client import HTTPMessage

import phishbench.input.url_input._url_data as url_data


class URLDataTest(unittest.TestCase):
    """
    The testcase for the `URLData` model
    """
    def test_is_ip_address(self):
        """
        Tests `url_data.is_ip_address` with an ip address URL
        """
        test_domain = '192.168.0.1'
        self.assertTrue(url_data.is_ip_address(test_domain))

    def test_is_not_ip_address(self):
        """
        Tests `url_data.is_ip_address` with an non-ip address URL
        """
        test_domain = 'google.com'
        self.assertFalse(url_data.is_ip_address(test_domain))

    def test_disable_download_url(self):
        """
        Tests to ensure that the download_url parameter of the `URLData` constructor works
        """
        test_url = 'http://google.com'
        data = url_data.URLData(test_url, download_url=False)
        self.assertIsNone(data.downloaded_website)
        self.assertIsNone(data.dns_results)
        self.assertIsNone(data.ip_whois)
        self.assertIsNone(data.domain_whois)

    def test_download_url(self):
        """
        Tests to ensure that downloading websites works
        """
        test_url = 'http://google.com'
        data = url_data.URLData(test_url, download_url=True)

        self.assertGreater(data.load_time, 0)
        self.assertIsNotNone(data.downloaded_website)
        self.assertIsInstance(data.downloaded_website, str)
        self.assertTrue("google.com" in data.final_url)
        self.assertIsNotNone(data.website_headers)
        self.assertIsInstance(data.website_headers, HTTPMessage)

    @patch('whois.whois')
    def test_download_whois_subdomain(self, whois_mock: MagicMock):
        """
        Tests to ensure that the fetching whois information works
        """
        # pylint: disable=no-self-use
        test_url = 'https://foodwishes.blogspot.com/2020/08/garlic-rice-roast-chicken-getting-under.html'
        whois_mock.return_value = {
            "domain_name": 'GOOGLE.COM',
            "org": "Google LLC"
        }
        data = url_data.URLData(test_url, download_url=False)
        data.lookup_whois()

        whois_mock.assert_called_once_with('blogspot.com')

    @patch('whois.whois')
    def test_download_whois(self, whois_mock: MagicMock):
        """
        Tests to ensure that the fetching whois information works
        """
        test_url = 'http://google.com'
        whois_mock.return_value = {
            "domain_name": 'GOOGLE.COM',
            "org": "Google LLC"
        }
        data = url_data.URLData(test_url, download_url=False)
        data.lookup_whois(nameservers=['1.1.1.1'])

        whois_mock.assert_called_once_with('google.com')
        whois_info = data.domain_whois

        self.assertEqual("GOOGLE.COM", whois_info['domain_name'])
        self.assertEqual("Google LLC", whois_info['org'])

        self.assertIsInstance(data.ip_whois, list)
        for x in data.ip_whois:
            self.assertIsInstance(x, dict)

    def test_lookup_dns(self):
        """
        Tests to ensure that looking up DNS information works
        """
        test_url = 'http://google.com/test?bacon=1'
        data = url_data.URLData(test_url, download_url=False)
        data.lookup_dns(nameservers=['1.1.1.1'])

        self.assertIsInstance(data.dns_results, dict)
        self.assertGreater(len(data.dns_results), 0)
