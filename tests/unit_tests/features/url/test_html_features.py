import unittest

from phishbench.feature_extraction.url.features import website_features
from phishbench.feature_extraction.url.features import website_html_features
from tests import mock_objects


class TestHTMLReflectionFeatures(unittest.TestCase):

    def test_is_redirect_true(self):
        # https://bit.ly/2Ef4uAS captured on 8/30/2020
        # redirects to https://en.wikipedia.org/wiki/Eastman_Kodak_Co._v._Image_Technical_Services,_Inc.
        test_url = mock_objects.get_mock_urldata('wikipedia_redirect')
        result = website_features.is_redirect(test_url)
        self.assertTrue(result)

    def test_is_redirect_false(self):
        # https://microsoft.com captured on 8/30/2020
        test_url = mock_objects.get_mock_urldata('microsoft')
        result = website_features.is_redirect(test_url)
        self.assertFalse(result)

    def test_content_type_encoding(self):
        test_url = mock_objects.get_mock_urldata('microsoft')
        test_url.website_headers['Content-Type'] = 'text/html; encoding=utf-8'

        result = website_features.content_type_header(test_url)
        self.assertEqual('text/html', result)

    def test_content_type_no_encoding(self):
        test_url = mock_objects.get_mock_urldata('microsoft')
        test_url.website_headers['Content-Type'] = 'text/html'
        result = website_features.content_type_header(test_url)
        self.assertEqual('text/html', result)

    def test_content_length(self):
        test_url = mock_objects.get_mock_urldata('microsoft')

        result = website_features.content_length_header(test_url)
        self.assertEqual(291, result)

    def test_x_powered_by(self):
        test_url = mock_objects.get_mock_urldata('microsoft')
        test_url.website_headers['X-Powered-By'] = "PHP/5.4.0"
        result = website_features.x_powered_by_header(test_url)
        self.assertIsInstance(result, str)
        self.assertEqual("PHP/5.4.0", result)

    def test_x_powered_by_empty(self):
        test_url = mock_objects.get_mock_urldata('microsoft')

        result = website_features.x_powered_by_header(test_url)
        self.assertIsInstance(result, str)
        self.assertEqual("", result)

    def test_number_of_tags(self):
        test_url = mock_objects.get_mock_urldata('microsoft')

        result = website_html_features.number_of_tags(test_url)
        self.assertEqual(6, result)

    def test_number_of_head(self):
        test_url = mock_objects.get_mock_urldata('microsoft')

        result = website_html_features.number_of_head(test_url)
        self.assertEqual(1, result)

    def test_number_of_html(self):
        test_url = mock_objects.get_mock_urldata('microsoft')

        result = website_html_features.number_of_html(test_url)
        self.assertEqual(1, result)

    def test_number_of_body(self):
        test_url = mock_objects.get_mock_urldata('microsoft')

        result = website_html_features.number_of_body(test_url)
        self.assertEqual(1, result)

    def test_number_of_title(self):
        test_url = mock_objects.get_mock_urldata('microsoft')

        result = website_html_features.number_of_title(test_url)
        self.assertEqual(1, result)

    def test_number_of_iframe_zero(self):
        test_url = mock_objects.get_mock_urldata('microsoft')

        result = website_html_features.number_of_iframe(test_url)
        self.assertEqual(0, result)

    def test_number_of_iframe(self):
        test_url = mock_objects.get_mock_urldata('iframe')

        result = website_html_features.number_of_iframe(test_url)
        self.assertEqual(3, result)

    def test_number_of_input(self):
        test_url = mock_objects.get_mock_urldata('wikipedia_redirect')

        result = website_html_features.number_of_input(test_url)
        self.assertEqual(7, result)

    def test_number_of_img(self):
        test_url = mock_objects.get_mock_urldata('wikipedia_redirect')

        result = website_html_features.number_of_img(test_url)
        self.assertEqual(6, result)

    def test_number_of_scripts(self):
        test_url = mock_objects.get_mock_urldata('wikipedia_redirect')

        result = website_html_features.number_of_script(test_url)
        self.assertEqual(6, result)

    def test_number_of_anchor(self):
        test_url = mock_objects.get_mock_urldata('wikipedia_redirect')

        result = website_html_features.number_of_anchor(test_url)
        self.assertEqual(343, result)

    def test_number_of_embed(self):
        test_url = mock_objects.get_mock_urldata('wikipedia_redirect')

        result = website_html_features.number_of_embed(test_url)
        self.assertEqual(0, result)

    def test_hidden_input(self):
        test_url = mock_objects.get_mock_urldata('wikipedia_redirect')
        test_url.downloaded_website = mock_objects.get_html("test_1.html")
        result = website_html_features.number_of_hidden_input(test_url)
        self.assertEqual(1, result)

    def test_hidden_svg(self):
        test_url = mock_objects.get_mock_urldata('wikipedia_redirect')
        test_url.downloaded_website = mock_objects.get_html("test_1.html")
        result = website_html_features.number_of_hidden_svg(test_url)
        self.assertEqual(1, result)
