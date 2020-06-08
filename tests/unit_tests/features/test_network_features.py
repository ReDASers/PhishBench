import unittest
from unittest.mock import patch

from phishbench import Features
from phishbench.input.url_input import URLData
from tests.mock_objects import mock_objects

@patch('phishbench.utils.Globals.config', new_callable=mock_objects.get_mock_config)
class TestNetworkFeatures(unittest.TestCase):

    def test_Network_number_name_server(self, mock_config):
        mock_config["Network_Features"]["number_name_server"] = "True"
        list_features = {}
        list_time = {}
        url_data = URLData('google.com', False)
        url_data.lookup_dns(nameservers=['1.1.1.1'])

        Features.Network_number_name_server(url_data.dns_results, list_features, list_time)
        print(url_data.dns_results)

        self.assertEqual(list_features["number_name_server"], 4, 'incorrect number_of_tags')

    # WHOIS test takes too much time on GitHub Actions
    # def test_whois_updated(self, mock_config):
    #     mock_config["Network_Features"]["updated_date"] = "True"
    #     list_features = {}
    #     list_time = {}
    #     url_data = URLData('google.com', False)
    #     url_data.lookup_whois(nameservers=['1.1.1.1'])
    #
    #     Features.Network_updated_date(url_data.domain_whois, list_features, list_time)
    #     print(list_features["updated_date"])
    #     self.assertGreater(list_features["updated_date"], 0,  'incorrect update date')