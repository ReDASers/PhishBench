"""
Tests JavaScript features
"""
import unittest

from phishbench.feature_extraction.url.features import javascript_features
from tests import mock_objects


# pylint: disable=missing-function-docstring
# pylint: disable=too-many-public-methods


class TesJavaScriptReflectionFeatures(unittest.TestCase):
    """
    Tests javascript_features
    """

    def test_num_exec(self):
        test_url = mock_objects.get_mock_object('regex_tester_urldata.pkl')

        result = javascript_features.number_of_exec().extract(test_url)
        self.assertEqual(1, result)

    def test_num_escape(self):
        test_url = mock_objects.get_mock_object('escape_urldata.pkl')
        result = javascript_features.number_of_escape().extract(test_url)
        self.assertEqual(2, result)

    def test_number_of_eval(self):
        test_url = mock_objects.get_mock_object('escape_urldata.pkl')
        result = javascript_features.number_of_eval().extract(test_url)
        self.assertEqual(6, result)

    def test_number_of_link(self):
        test_url = mock_objects.get_mock_object('escape_urldata.pkl')
        result = javascript_features.number_of_link().extract(test_url)
        self.assertEqual(3, result)

    def test_num_unescape(self):
        test_url = mock_objects.get_mock_object('escape_urldata.pkl')
        result = javascript_features.number_of_unescape().extract(test_url)
        self.assertEqual(1, result)

    def test_num_search(self):
        test_url = mock_objects.get_mock_object('escape_urldata.pkl')
        result = javascript_features.number_of_search().extract(test_url)
        self.assertEqual(2, result)

    def test_num_set_timeout(self):
        test_url = mock_objects.get_mock_object('escape_urldata.pkl')
        result = javascript_features.number_of_set_timeout().extract(test_url)
        self.assertEqual(4, result)

    def test_num_iframe(self):
        test_url = mock_objects.get_mock_object('escape_urldata.pkl')
        result = javascript_features.number_of_iframes_in_script().extract(test_url)
        self.assertEqual(2, result)

    def test_right_click_modified(self):
        test_url = mock_objects.get_mock_object('right_click_disabled.pkl')
        result = javascript_features.right_click_modified().extract(test_url)
        self.assertTrue(result)

    def test_number_of_event_attachment(self):
        test_url = mock_objects.get_mock_object('events_urldata.pkl')
        result = javascript_features.number_of_event_attachment().extract(test_url)
        self.assertEqual(3, result)

    def test_number_of_event_dispatch(self):
        test_url = mock_objects.get_mock_object('events_urldata.pkl')
        result = javascript_features.number_of_event_dispatch().extract(test_url)
        self.assertEqual(3, result)
