import unittest
from unittest.mock import patch

from phishbench import Features
import phishbench.feature_extraction.url.features as url_features
from phishbench.input import URLData
from tests.mock_objects import mock_objects


@patch('phishbench.utils.phishbench_globals.config', new_callable=mock_objects.get_mock_config)
class TestURLFeatures(unittest.TestCase):

    def test_URL_url_length(self, config_mock):
        test_url = 'http://te2t-url.com/home.html'
        test_url = URLData(test_url, download_url=False)

        result = url_features.url_length(test_url)

        self.assertEqual(29, result)

    def test_URL_domain_length(self, config_mock):
        test_url = 'http://www.google.com/index.html'
        test_url = URLData(test_url, download_url=False)

        result = url_features.domain_length(test_url)

        self.assertEqual(result, 14)

    def test_URL_letter_occurrence(self, config_mock):
        config_mock['URL_Features']["letter_occurrence"] = "True"
        list_features = {}
        list_time = {}

        Features.URL_letter_occurrence('http://te2t-url.com/home.html', list_features, list_time)

        self.assertEqual(list_features["letter_occurrence_t"], 4, 'incorrect letter_occurrence')
        self.assertEqual(list_features["letter_occurrence_c"], 1, 'incorrect letter_occurrence')

    def test_URL_char_distance(self, config_mock):
        pass

    def test_URL_kolmogorov_shmirnov(self, config_mock):
        pass

    def test_URL_Kullback_Leibler_Divergence(self, config_mock):
        pass

    def test_URL_english_frequency_distance(self, config_mock):
        pass

    def test_URL_num_punctuation(self, config_mock):
        config_mock['URL_Features']['num_punctuation'] = "True"
        list_features = {}
        list_time = {}

        Features.URL_num_punctuation('http://te2t-url.com/home.html', list_features, list_time)

        self.assertEqual(list_features["num_punctuation"], 7, 'incorrect num_punctuation')

    def test_URL_has_port(self, config_mock):
        test_url = URLData('http://www.google.com:443', download_url=False)
        self.assertTrue(url_features.has_port(test_url))

        test_url = URLData('http://www.google.com', download_url=False)
        self.assertFalse(url_features.has_port(test_url))

    def test_URL_number_of_digits(self, config_mock):
        url = URLData('http://te2t-url.com/home.html', download_url=False)
        result = url_features.num_digits(url)

        self.assertEqual(result, 1, 'incorrect number_of_digits')

        url = URLData('http://te2t-url.com/home42.html', download_url=False)
        result = url_features.num_digits(url)

        self.assertEqual(result, 3, 'incorrect number_of_digits')

    def test_URL_number_of_dots(self, config_mock):
        url = URLData('http://te2t-url.com/home.html', download_url=False)
        result = url_features.num_dots(url)

        self.assertEqual(result, 2, 'incorrect number_of_dots')

        url = URLData('http://test.te2t-url.com/home.html', download_url=False)
        result = url_features.num_dots(url)

        self.assertEqual(result, 3, 'incorrect number_of_dots')

    def test_URL_number_of_slashes(self, config_mock):
        config_mock['URL_Features']['number_of_slashes'] = "True"
        list_features = {}
        list_time = {}

        Features.URL_number_of_slashes('http://te2t-url.com/home.html', list_features, list_time)

        self.assertEqual(list_features["number_of_slashes"], 3, 'incorrect number_of_slashes')

    def test_URL_digit_letter_ratio(self, config_mock):
        config_mock['URL_Features']['digit_letter_ratio'] = "True"
        list_features = {}
        list_time = {}

        Features.URL_digit_letter_ratio('http://te2t-url.com/home.html', list_features, list_time)

        self.assertEqual(list_features["digit_letter_ratio"], 1 / 21, 'incorrect digit_letter_ratio')

    # def test_URL_consecutive_numbers(self, config_mock):
    #     config_mock['URL_Features']['consecutive_numbers'] = "True"
    #     list_features = {}
    #     list_time = {}
    #
    #     Features.URL_consecutive_numbers('http://te2t-url.com/home.html', list_features, list_time)
    #
    #     self.assertEqual(list_features["consecutive_numbers"], 1, 'incorrect consecutive_numbers')

    def test_URL_special_char_count(self, config_mock):
        config_mock['URL_Features']['special_char_count'] = "True"
        list_features = {}
        list_time = {}

        Features.URL_special_char_count('http://te2t-url.com/home.html', list_features, list_time)

        self.assertEqual(list_features["special_char_count"], 1, 'incorrect special_char_count')

    # def test_URL_special_pattern(self, config_mock):
    #     config_mock['URL_Features']['special_pattern'] = "True"
    #     test_url = 'https://www.google.com/?gws_rd=ssl'
    #     list_features = {}
    #     list_time = {}
    #
    #     Features.URL_special_pattern(test_url, list_features, list_time)
    #
    #     self.assertEqual(list_features["special_pattern"], 1, 'incorrect special_pattern')

    def test_URL_Top_level_domain(self, config_mock):
        config_mock['URL_Features']['Top_level_domain'] = "True"
        list_features = {}
        list_time = {}

        Features.URL_Top_level_domain('http://te2t-url.com/home.html', list_features, list_time)

        self.assertEqual(list_features["Top_level_domain"], 'com', 'incorrect Top_level_domain')

    def test_url_is_common_tld(self, config_mock):
        test_url = URLData('http://www.test.co.uk', download_url=False)

        result = url_features.is_common_tld(test_url)
        self.assertFalse(result)

        test_url = URLData("http://www.google.com", download_url=False)
        result = url_features.is_common_tld(test_url)
        self.assertTrue(result)

    def test_url_is_ip_addr(self, config_mock):
        test_url = URLData('http://192.168.0.1/scripts/test', download_url=False)

        result = url_features.is_ip_addr(test_url)
        self.assertTrue(result)

        test_url = URLData('http://www.google.com/scripts/test', download_url=False)

        result = url_features.is_ip_addr(test_url)
        self.assertFalse(result)

    def test_url_has_https(self, config_mock):
        test_url = URLData('https://google.com', download_url=False)
        self.assertTrue(url_features.has_https(test_url))

        test_url = URLData('http://google.com', download_url=False)
        self.assertFalse(url_features.has_https(test_url))

    def test_URL_number_of_dashes(self, config_mock):

        test_url = URLData('http://te2t-ur--l.com/home.html', download_url=False)
        self.assertEqual(3, url_features.number_of_dashes(test_url), 'incorrect number_of_dashes')

    def test_URL_Http_middle_of_URL_true_no_start(self, config_mock):
        test_url = URLData("google.com?https://google.com", download_url=False)
        self.assertTrue(url_features.http_in_middle(test_url))

        test_url = URLData("http://www.google.com", download_url=False)
        self.assertFalse(url_features.http_in_middle(test_url))

    def test_URL_Has_More_than_3_dots(self, config_mock):
        config_mock['URL_Features']['Has_More_than_3_dots'] = "True"
        test_url = "http://tech.www.google.com/index.html"
        list_features = {}
        list_time = {}

        Features.URL_Has_More_than_3_dots(test_url, list_features, list_time)

        self.assertEqual(list_features["Has_More_than_3_dots"], 1, 'incorrect Http_middle_of_URL')

    def test_URL_Has_at_symbole(self, config_mock):
        test_url = "http://tech.www.google.com/index.html@http"

        self.assertTrue(url_features.has_at_symbol(URLData(test_url, download_url=False)))
        test_url = "http://tech.www.google.com/index.html"
        self.assertFalse(url_features.has_at_symbol(URLData(test_url, download_url=False)))

    def test_URL_Has_anchor_tag_true(self, config_mock):
        config_mock['URL_Features']['Has_anchor_tag'] = "True"
        test_url = "anchor_example2.html#a001"
        list_features = {}
        list_time = {}

        Features.URL_Has_anchor_tag(test_url, list_features, list_time)

        self.assertEqual(list_features["Has_anchor_tag"], 1, 'incorrect Has_anchor_tag')

    def test_URL_Has_anchor_tag_false(self, config_mock):
        config_mock['URL_Features']['Has_anchor_tag'] = "True"
        test_url = "anchor_example2.html"
        list_features = {}
        list_time = {}

        Features.URL_Has_anchor_tag(test_url, list_features, list_time)

        self.assertEqual(list_features["Has_anchor_tag"], 0, 'incorrect Has_anchor_tag')

    def test_URL_Null_in_Domain(self, config_mock):

        test_url = URLData('http://www.test.com/test/null', download_url=False)
        self.assertFalse(url_features.null_in_domain(test_url))

        test_url = URLData('http://www.nulltest.com/test', download_url=False)
        self.assertTrue(url_features.null_in_domain(test_url))


    def test_URL_Token_Count(self, config_mock):
        config_mock['URL_Features']['Token_Count'] = "True"
        list_features = {}
        list_time = {}

        Features.URL_Token_Count('http://te2t-url.com/home.html', list_features, list_time)

        self.assertEqual(6, list_features["Token_Count"], 'incorrect Token_Count')

    def test_URL_Average_Path_Token_Length(self, config_mock):
        config_mock['URL_Features']['Average_Path_Token_Length'] = "True"
        list_features = {}
        list_time = {}

        Features.URL_Average_Path_Token_Length('http://te2t-url.com/home.html', list_features, list_time)

        self.assertEqual(list_features["Average_Path_Token_Length"], 8 / 3, 'incorrect Average_Path_Token_Length')

    def test_URL_Average_Domain_Token_Length(self, config_mock):
        config_mock['URL_Features']['Average_Domain_Token_Length'] = "True"
        list_features = {}
        list_time = {}

        Features.URL_Average_Domain_Token_Length('http://te2t-url.com/home.html', list_features, list_time)

        self.assertEqual(list_features["Average_Domain_Token_Length"], 10 / 3, 'incorrect Average_Domain_Token_Length')

    def test_URL_Longest_Domain_Token(self, config_mock):
        config_mock['URL_Features']['Longest_Domain_Token'] = "True"
        list_features = {}
        list_time = {}

        Features.URL_Longest_Domain_Token('http://te2t-url.com/home.html', list_features, list_time)

        self.assertEqual(4, list_features["Longest_Domain_Token"], 'incorrect Longest_Domain_Token')

    def test_URL_Protocol_Port_Match(self, config_mock):
        config_mock['URL_Features']['Protocol_Port_Match'] = "True"
        list_features = {}
        list_time = {}

        Features.URL_Protocol_Port_Match('http://te2t-url.com/home.html', list_features, list_time)

        self.assertEqual(list_features["Protocol_Port_Match"], 1, 'incorrect Protocol_Port_Match')

    def test_URL_Has_WWW_in_Middle(self, config_mock):
        config_mock['URL_Features']['Has_WWW_in_Middle'] = "True"
        test_url = "http://httpwwwchase.com"
        list_features = {}
        list_time = {}

        Features.URL_Has_WWW_in_Middle(test_url, list_features, list_time)

        self.assertEqual(list_features["Has_WWW_in_Middle"], 1, 'incorrect Has_WWW_in_Middle')

    def test_URL_Has_Hex_Characters(self, config_mock):
        config_mock['URL_Features']['Has_Hex_Characters'] = "True"
        test_url = 'text=Hello+G%C3%BCnterasd'
        list_features = {}
        list_time = {}

        Features.URL_Has_Hex_Characters(test_url, list_features, list_time)

        self.assertEqual(list_features["Has_Hex_Characters"], 1, 'incorrect Has_Hex_Characters')

    def test_URL_Double_Slashes_Not_Beginning_Count(self, config_mock):
        config_mock['URL_Features']['Double_Slashes_Not_Beginning_Count'] = "True"
        test_url = 'http://ff.com//s//sss//'
        list_features = {}
        list_time = {}

        Features.URL_Double_Slashes_Not_Beginning_Count(test_url, list_features, list_time)

        self.assertEqual(list_features["Double_Slashes_Not_Beginning_Count"], 1,
                         'incorrect Double_Slashes_Not_Beginning_Count')

    def test_URL_Brand_In_Url(self, config_mock):
        config_mock['URL_Features']['Brand_In_URL'] = "True"
        test_url = 'http://ff.com/microsoft.com//'
        list_features = {}
        list_time = {}

        Features.URL_Brand_In_Url(test_url, list_features, list_time)

        self.assertEqual(list_features["Brand_In_URL"], 1, 'incorrect Brand_In_URL')

    def test_URL_Is_Whitelisted(self, config_mock):
        config_mock['URL_Features']['Is_Whitelisted'] = "True"
        test_url = 'http://microsoft.com/microsoft.com//'
        list_features = {}
        list_time = {}

        Features.URL_Is_Whitelisted(test_url, list_features, list_time)

        self.assertEqual(list_features["Is_Whitelisted"], 1, 'incorrect Is_Whitelisted')

