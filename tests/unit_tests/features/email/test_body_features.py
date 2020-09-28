"""
Tests for email body features
"""
import unittest

from phishbench.feature_extraction.email.features import email_body
from phishbench.input.email_input.models import EmailBody
from tests.sample_emails import utils  # pylint: disable=import-error


class TestBodyFeatures(unittest.TestCase):
    """
    Tests internal body features
    """
    def test_blacklist_words_body(self):
        msg = utils.get_binary_email('HeaderTests/Test Email 1.txt')
        body = EmailBody(msg)
        body.text = "This is an important message from PayPal. Please verify your paypal account now"
        result = email_body.blacklisted_words_body().extract(body)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['paypal'], 2)
        self.assertEqual(result['account'], 1)
