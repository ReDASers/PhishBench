import unittest
from unittest.mock import patch, MagicMock

import phishbench.input.url_input._url_data as url_data


class Test(unittest.TestCase):

    def test_is_ip_address(self):
        test_domain = '192.168.0.1'
        self.assertTrue(url_data.is_ip_address(test_domain))

    def test_is_not_ip_address(self):
        test_domain = 'google.com'
        self.assertFalse(url_data.is_ip_address(test_domain))

    def test_disable_download_url(self):
        test_url = 'http://google.com'
        data = url_data.URLData(test_url, download_url=False)
        self.assertIsNone(data.downloaded_website)
        self.assertIsNone(data.dns_results)
        self.assertIsNone(data.ip_whois)
        self.assertIsNone(data.domain_whois)

    def test_download_url(self):
        test_url = 'http://google.com'
        data = url_data.URLData(test_url, download_url=True)

        self.assertGreater(data.load_time, 0)
        self.assertIsNotNone(data.downloaded_website)
        self.assertIsInstance(data.downloaded_website, str)
        self.assertTrue("google.com" in data.final_url)
        self.assertIsNotNone(data.website_headers)

    @patch('whois.whois')
    def test_download_whois(self, whois_mock: MagicMock):
        test_url = 'http://google.com'
        whois_mock.return_value = {
            "domain_name": 'GOOGLE.COM',
            "org": "Google LLC"
        }
        data = url_data.URLData(test_url, download_url=False)
        data.lookup_whois()

        whois_mock.assert_called_once_with('google.com')
        whois_info = data.domain_whois

        self.assertEqual("GOOGLE.COM", whois_info['domain_name'])
        self.assertEqual("Google LLC", whois_info['org'])

        self.assertGreater(len(data.ip_whois), 0)
        self.assertEqual('GOOGLE, US', data.ip_whois[0]['asn_description'])

    def test_lookup_dns(self):
        test_url = 'http://google.com/test?bacon=1'
        data = url_data.URLData(test_url, download_url=False)
        data.lookup_dns(nameservers=['1.1.1.1'])
        self.assertTrue(data.dns_results)

    # WHOIS test takes too much time on GitHub Actions
    # def test_lookup_whois(self):
    #     test_url = 'http://google.com/test?bacon=1'
    #     data = url_data.URLData(test_url, download_url=False)
    #     data.lookup_whois(nameservers=['1.1.1.1'])
    #     self.assertIsNotNone(data.domain_whois)
    #     self.assertEqual('GOOGLE, US', data.ip_whois[0]['asn_description'])
