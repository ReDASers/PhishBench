import unittest

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

    def test_URLData_domaino(self):
        test_url = 'http://google.com/test?bacon=1'
        data = url_data.URLData(test_url, download_url=False)
        self.assertEqual('google.com', data.domain)

    def test_URLData_path(self):
        test_url = 'http://google.com/test?bacon=1'
        data = url_data.URLData(test_url, download_url=False)
        self.assertEqual('/test', data.path)

    def test_URLData_query(self):
        test_url = 'http://google.com/test?bacon=1'
        data = url_data.URLData(test_url, download_url=False)
        self.assertEqual('bacon=1', data.query)

    def test_lookup_dns(self):
        test_url = 'http://google.com/test?bacon=1'
        data = url_data.URLData(test_url, download_url=False)
        data.lookup_dns(nameservers=['1.1.1.1'])
        self.assertTrue(data.dns_results)

    def test_lookup_whois(self):
        test_url = 'http://google.com/test?bacon=1'
        data = url_data.URLData(test_url, download_url=False)
        data.lookup_whois(nameservers=['1.1.1.1'])
        self.assertIsNotNone(data.domain_whois)
        self.assertEqual('GOOGLE, US', data.ip_whois[0]['asn_description'])
