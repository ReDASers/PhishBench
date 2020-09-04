"""
Tests for raw url features
"""
import unittest

import phishbench.feature_extraction.url.features as url_features
from phishbench.input import URLData


# pylint: disable=missing-function-docstring
# pylint: disable=too-many-public-methods


class TestURLReflectionFeatures(unittest.TestCase):
    """
    Tests raw_url_features
    """

    def test_url_length(self):
        test_url = 'http://te2t-url.com/home.html'
        test_url = URLData(test_url, download_url=False)

        result = url_features.url_length(test_url)

        self.assertEqual(29, result)

    def test_domain_length(self):
        test_url = 'http://www.google.com/index.html'
        test_url = URLData(test_url, download_url=False)

        result = url_features.domain_length(test_url)

        self.assertEqual(result, 14)

    def test_char_distance(self):
        test_url = URLData('http://te2t-url.com/home.html', download_url=False)

        result = url_features.char_dist(test_url)
        self.assertEqual(26, len(result))

    def test_num_punctuation(self):
        test_url = URLData('http://te2t-url.com/home.html', download_url=False)

        self.assertEqual(7, url_features.num_punctuation(test_url), 'incorrect num_punctuation')

    def test_has_port(self):
        test_url = URLData('http://www.google.com:443', download_url=False)
        self.assertTrue(url_features.has_port(test_url))

        test_url = URLData('http://www.google.com', download_url=False)
        self.assertFalse(url_features.has_port(test_url))

    def test_number_of_digits(self):
        url = URLData('http://te2t-url.com/home.html', download_url=False)
        result = url_features.num_digits(url)

        self.assertEqual(result, 1, 'incorrect number_of_digits')

        url = URLData('http://te2t-url.com/home42.html', download_url=False)
        result = url_features.num_digits(url)

        self.assertEqual(result, 3, 'incorrect number_of_digits')

    def test_number_of_dots(self):
        url = URLData('http://te2t-url.com/home.html', download_url=False)
        result = url_features.num_dots(url)

        self.assertEqual(result, 2, 'incorrect number_of_dots')

        url = URLData('http://test.te2t-url.com/home.html', download_url=False)
        result = url_features.num_dots(url)

        self.assertEqual(result, 3, 'incorrect number_of_dots')

    def test_number_of_slashes(self):
        test_url = URLData('http://te2t-url.com\\home.html', download_url=False)
        result = url_features.number_of_slashes(test_url)

        self.assertEqual(3, result)

    def test_digit_letter_ratio(self):
        test_url = URLData('http://te2t-url.com/home.html', download_url=False)
        result = url_features.digit_letter_ratio(test_url)

        self.assertEqual(1 / 21, result, 'incorrect digit_letter_ratio')

    def test_url_is_common_tld(self):
        test_url = URLData('http://www.test.co.uk', download_url=False)

        result = url_features.is_common_tld(test_url)
        self.assertFalse(result)

        test_url = URLData("http://www.google.com", download_url=False)
        result = url_features.is_common_tld(test_url)
        self.assertTrue(result)

    def test_url_is_ip_addr(self):
        test_url = URLData('http://192.168.0.1/scripts/test', download_url=False)

        result = url_features.is_ip_addr(test_url)
        self.assertTrue(result)

        test_url = URLData('http://www.google.com/scripts/test', download_url=False)

        result = url_features.is_ip_addr(test_url)
        self.assertFalse(result)

    def test_url_has_https(self):
        test_url = URLData('https://google.com', download_url=False)
        self.assertTrue(url_features.has_https(test_url))

        test_url = URLData('http://google.com', download_url=False)
        self.assertFalse(url_features.has_https(test_url))

    def test_number_of_dashes(self):
        test_url = URLData('http://te2t-ur--l.com/home.html', download_url=False)
        self.assertEqual(3, url_features.number_of_dashes(test_url), 'incorrect number_of_dashes')

    def test_http_middle_of_true_no_start(self):
        test_url = URLData("google.com?https://google.com", download_url=False)
        self.assertTrue(url_features.http_in_middle(test_url))

        test_url = URLData("http://www.google.com", download_url=False)
        self.assertFalse(url_features.http_in_middle(test_url))

    def test_has_at_symbol(self):
        test_url = "http://tech.www.google.com/index.html@http"

        self.assertTrue(url_features.has_at_symbol(URLData(test_url, download_url=False)))
        test_url = "http://tech.www.google.com/index.html"
        self.assertFalse(url_features.has_at_symbol(URLData(test_url, download_url=False)))

    def test_null_in_domain(self):
        test_url = URLData('http://www.test.com/test/null', download_url=False)
        self.assertFalse(url_features.null_in_domain(test_url))

        test_url = URLData('http://www.nulltest.com/test', download_url=False)
        self.assertTrue(url_features.null_in_domain(test_url))

    def test_special_pattern(self):
        test_url = URLData('http://www.test.com/test/null', download_url=False)
        self.assertFalse(url_features.special_pattern(test_url))

        test_url = URLData('https://www.google.com/?gws_rd=ssl', download_url=False)
        self.assertTrue(url_features.special_pattern(test_url))

    def test_consecutive_numbers(self):
        test_url = URLData('http://abc123-45659.com/home22.html', download_url=False)

        result = url_features.consecutive_numbers(test_url)

        self.assertEqual(38, result, 'incorrect consecutive_numbers')

    def test_top_level_domain(self):
        test_url = URLData('http://abc123-45659.com/home22.html', download_url=False)
        result = url_features.top_level_domain(test_url)

        self.assertEqual('com', result, 'incorrect Top_level_domain')

    def test_special_char_count(self):
        test_url = URLData('http://te2t-url.com/h@ome.html', download_url=False)
        result = url_features.special_char_count(test_url)
        self.assertEqual(2, result)

    def test_has_more_than_three_dots_true(self):
        test_url = URLData("http://www.tec.h.google.com/index.html", download_url=False)

        result = url_features.has_more_than_three_dots(test_url)
        self.assertTrue(result)

    def test_has_more_than_three_dots_false(self):
        test_url = URLData("http://www.tech.google.com/index.html", download_url=False)

        result = url_features.has_more_than_three_dots(test_url)
        self.assertFalse(result)

    def test_has_anchor_tag_true(self):
        test_url = URLData("http://www.anchor_example2.com/x.html#a001", download_url=False)

        result = url_features.has_anchor_tag(test_url)
        self.assertTrue(result)

    def test_has_anchor_tag_false(self):
        test_url = URLData("http://www.anchor_example2.com/x.html", download_url=False)

        result = url_features.has_anchor_tag(test_url)
        self.assertFalse(result)

    def test_letter_occurrence(self):
        test_url = URLData('http://te2t-url.com/home.html', download_url=False)

        result = url_features.domain_letter_occurrence(test_url)
        self.assertEqual(26, len(result))
        self.assertEqual(0, result['domain_letter_occurrence_h'])
        self.assertEqual(2, result['domain_letter_occurrence_t'])

    def test_has_hex_characters_true(self):
        test_url = URLData('http://te2t-url.com/index.html?text=Hello+G%C3%BCnterasd', download_url=False)
        result = url_features.has_hex_characters(test_url)

        self.assertTrue(result)

    def test_has_hex_characters_false(self):
        test_url = URLData('http://te2t-url.com/index.html', download_url=False)

        result = url_features.has_hex_characters(test_url)

        self.assertFalse(result)

    def test_brand_in_url(self):
        test_url = URLData('http://ff.com/microsoft.com//', download_url=False)

        result = url_features.brand_in_url(test_url)

        self.assertTrue(result)

    def test_is_whitelisted_true(self):
        test_url = URLData('http://microsoft.com/index.html', download_url=False)

        result = url_features.is_whitelisted(test_url)

        self.assertTrue(result)

    def test_is_whitelisted_false(self):
        test_url = URLData('http://ff.com/microsoft.com//', download_url=False)

        result = url_features.is_whitelisted(test_url)

        self.assertFalse(result)

    def test_double_slashes_in_path(self):
        test_url = URLData('http://ff.com/s//sss//', download_url=False)

        result = url_features.double_slashes_in_path(test_url)
        self.assertEqual(2, result)

    def test_has_www_in_middle_false(self):
        test_url = URLData('http://www.google.com', download_url=False)
        result = url_features.has_www_in_middle(test_url)
        self.assertFalse(result)

    def test_has_www_in_middle_domain(self):
        test_url = URLData('http://www.google.com.www.test.com', download_url=False)
        result = url_features.has_www_in_middle(test_url)
        self.assertTrue(result)

    def test_has_www_in_middle_path(self):
        test_url = URLData('http://www.google.com/www.test.com', download_url=False)
        result = url_features.has_www_in_middle(test_url)
        self.assertTrue(result)

    def test_port_protocol_match_no_port(self):
        test_url = URLData('http://www.google.com/www.test.com', download_url=False)
        result = url_features.protocol_port_match(test_url)
        self.assertTrue(result)

    def test_port_protocol_match(self):
        test_url = URLData('https://www.google.com:443/abc', download_url=False)
        result = url_features.protocol_port_match(test_url)
        self.assertTrue(result)

    def test_port_protocol_match_false(self):
        test_url = URLData('https://www.google.com:8080/abc', download_url=False)
        result = url_features.protocol_port_match(test_url)
        self.assertFalse(result)

    def test_port_protocol_match_unknown(self):
        test_url = URLData('stratum+tcp://scrypt.LOCATION.nicehash.com:3333', download_url=False)
        result = url_features.protocol_port_match(test_url)
        self.assertFalse(result)


class TestURLTokenFeatures(unittest.TestCase):
    """
    Tests url_token_features
    """

    def test_token_count(self):
        test_url = URLData('http://te2t-url.com/home.html', download_url=False)
        result = url_features.token_count(test_url)

        self.assertEqual(6, result)

    def test_average_path_token_length(self):
        test_url = URLData('http://te2t-url.com/home.html', download_url=False)
        result = url_features.average_path_token_length(test_url)

        self.assertEqual(4, result)

    def test_average_path_token_length_no_path(self):
        test_url = URLData('http://te2t-url.com', download_url=False)
        result = url_features.average_path_token_length(test_url)

        self.assertEqual(0, result)

    def test_average_domain_token_length(self):
        test_url = URLData('http://te2t-url.com/home.html', download_url=False)
        result = url_features.average_domain_token_length(test_url)

        self.assertEqual(10 / 3, result)

    def test_longest_domain_token_length(self):
        test_url = URLData('http://te2t-url.com/home.html', download_url=False)
        result = url_features.longest_domain_token_length(test_url)

        self.assertEqual(4, result)
