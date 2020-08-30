"""
Tests HTML features
"""
import unittest

from phishbench.feature_extraction.url.features import website_features
from phishbench.feature_extraction.url.features import website_html_features
from tests import mock_objects


# pylint: disable=missing-function-docstring
# pylint: disable=too-many-public-methods


class TestHTMLReflectionFeatures(unittest.TestCase):
    """
    Tests website_html_features
    """

    def test_is_redirect_true(self):
        # https://bit.ly/2Ef4uAS captured on 8/30/2020
        # redirects to https://en.wikipedia.org/wiki/Eastman_Kodak_Co._v._Image_Technical_Services,_Inc.
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')
        result = website_features.is_redirect().extract(test_url)
        self.assertTrue(result)

    def test_is_redirect_false(self):
        # https://microsoft.com captured on 8/30/2020
        test_url = mock_objects.get_mock_object('microsoft_urldata')

        result = website_features.is_redirect().extract(test_url)

        self.assertFalse(result)

    def test_content_type_encoding(self):
        test_url = mock_objects.get_mock_object('microsoft_urldata')
        test_url.website_headers['Content-Type'] = 'text/html; encoding=utf-8'

        result = website_features.content_type_header().extract(test_url)

        self.assertEqual('text/html', result)

    def test_content_type_no_encoding(self):
        test_url = mock_objects.get_mock_object('microsoft_urldata')
        test_url.website_headers['Content-Type'] = 'text/html'

        result = website_features.content_type_header().extract(test_url)

        self.assertEqual('text/html', result)

    def test_content_length(self):
        test_url = mock_objects.get_mock_object('microsoft_urldata')

        result = website_features.content_length_header().extract(test_url)

        self.assertEqual(291, result)

    def test_x_powered_by(self):
        test_url = mock_objects.get_mock_object('microsoft_urldata')
        test_url.website_headers['X-Powered-By'] = "PHP/5.4.0"

        result = website_features.x_powered_by_header().extract(test_url)

        self.assertIsInstance(result, str)
        self.assertEqual("PHP/5.4.0", result)

    def test_x_powered_by_empty(self):
        test_url = mock_objects.get_mock_object('microsoft_urldata')

        result = website_features.x_powered_by_header().extract(test_url)

        self.assertIsInstance(result, str)
        self.assertEqual("", result)

    def test_number_of_tags(self):
        test_url = mock_objects.get_mock_object('microsoft_urldata')

        result = website_html_features.number_of_tags().extract(test_url)
        self.assertEqual(6, result)

    def test_number_of_head(self):
        test_url = mock_objects.get_mock_object('microsoft_urldata')

        result = website_html_features.number_of_head().extract(test_url)
        self.assertEqual(1, result)

    def test_number_of_html(self):
        test_url = mock_objects.get_mock_object('microsoft_urldata')

        result = website_html_features.number_of_html().extract(test_url)
        self.assertEqual(1, result)

    def test_number_of_body(self):
        test_url = mock_objects.get_mock_object('microsoft_urldata')

        result = website_html_features.number_of_body().extract(test_url)
        self.assertEqual(1, result)

    def test_number_of_title(self):
        test_url = mock_objects.get_mock_object('microsoft_urldata')

        result = website_html_features.number_of_title().extract(test_url)
        self.assertEqual(1, result)

    def test_number_of_iframe_zero(self):
        test_url = mock_objects.get_mock_object('microsoft_urldata')

        result = website_html_features.number_of_iframe().extract(test_url)
        self.assertEqual(0, result)

    def test_number_of_iframe(self):
        test_url = mock_objects.get_mock_object('iframe_urldata')

        result = website_html_features.number_of_iframe().extract(test_url)
        self.assertEqual(3, result)

    def test_number_of_input(self):
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')

        result = website_html_features.number_of_input().extract(test_url)
        self.assertEqual(7, result)

    def test_number_of_img(self):
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')

        result = website_html_features.number_of_img().extract(test_url)
        self.assertEqual(6, result)

    def test_number_of_scripts(self):
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')

        result = website_html_features.number_of_script().extract(test_url)
        self.assertEqual(6, result)

    def test_number_of_anchor(self):
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')

        result = website_html_features.number_of_anchor().extract(test_url)
        self.assertEqual(343, result)

    def test_number_of_embed(self):
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')

        result = website_html_features.number_of_embed().extract(test_url)
        self.assertEqual(0, result)

    def test_hidden_input(self):
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')
        test_url.downloaded_website = mock_objects.get_webpage("test_1.html")
        result = website_html_features.number_of_hidden_input().extract(test_url)
        self.assertEqual(1, result)

    def test_hidden_svg(self):
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')
        test_url.downloaded_website = mock_objects.get_webpage("test_1.html")
        result = website_html_features.number_of_hidden_svg().extract(test_url)
        self.assertEqual(1, result)

    def test_number_external_content(self):
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')
        result = website_html_features.number_of_external_content().extract(test_url)

        self.assertEqual(3, result)

    def test_number_internal_content(self):
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')
        result = website_html_features.number_of_internal_content().extract(test_url)

        self.assertEqual(4, result)

    def test_number_internal_links(self):
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')
        result = website_html_features.number_of_internal_links().extract(test_url)

        self.assertEqual(321, result)

    def test_number_external_links(self):
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')
        result = website_html_features.number_of_external_links().extract(test_url)

        self.assertEqual(46, result)

    def test_number_suspicious_content(self):
        test_url = mock_objects.get_mock_object('wikipedia_shortener_urldata')
        result = website_html_features.number_suspicious_content().extract(test_url)

        self.assertEqual(145, result)

    def test_has_password_input(self):
        test_url = mock_objects.get_mock_object('github_login_urldata')
        result = website_html_features.has_password_input().extract(test_url)

        self.assertTrue(result)

    def test_has_password_input_false(self):
        """
        Tests on a website without a password input. Also tests against input without a type attribute
        """
        test_url = mock_objects.get_mock_object('google_urldata')
        result = website_html_features.has_password_input().extract(test_url)

        self.assertFalse(result)
