"""
Tests built-in email header features
"""
import unittest

from phishbench.feature_extraction.email.features import email_header
from phishbench.input.email_input.models import EmailHeader
from tests.sample_emails import utils  # pylint: disable=import-error

# pylint: disable=no-value-for-parameter
# pylint: disable=missing-function-docstring


class TestHeaderFeatures(unittest.TestCase):
    """
    Tests `email_header`
    """
    def test_mime_version(self):
        msg = utils.get_binary_email('HeaderTests/Test Email 1.txt')
        header = EmailHeader(msg)
        result = email_header.mime_version().extract(header)
        self.assertEqual("1.0", result)

    def test_size_in_bytes(self):
        msg = utils.get_binary_email('HeaderTests/Test Email 1.txt')
        header = EmailHeader(msg)
        result = email_header.size_in_bytes().extract(header)
        self.assertEqual(2957, result)

    def test_return_path(self):
        msg = utils.get_binary_email('HeaderTests/Test Email 1.txt')
        header = EmailHeader(msg)
        result = email_header.return_path().extract(header)
        self.assertEqual("return@domain.com", result)

    def test_x_mailer(self):
        msg = utils.get_binary_email('HeaderTests/Test Email 1.txt')
        header = EmailHeader(msg)
        header.x_mailer = 'TEST_MAILER'
        result = email_header.x_mailer().extract(header)
        self.assertEqual("TEST_MAILER", result)

    def test_x_originating_hostname(self):
        msg = utils.get_binary_email('HeaderTests/Test Email 3.txt')
        header = EmailHeader(msg)
        header.x_originating_hostname = 'abc.com'
        result = email_header.x_originating_hostname().extract(header)
        self.assertEqual('abc.com', result)

    def test_x_originating_ip(self):
        msg = utils.get_binary_email('HeaderTests/Test Email 3.txt')
        header = EmailHeader(msg)
        header.x_originating_ip = '123.255.255.255'
        result = email_header.x_originating_ip().extract(header)
        self.assertEqual('123.255.255.255', result)

    def test_x_virus_scanned(self):
        msg = utils.get_binary_email('HeaderTests/Test Email 3.txt')
        header = EmailHeader(msg)
        result = email_header.x_virus_scanned().extract(header)
        self.assertFalse(result)

    def test_x_spam_flag(self):
        msg = utils.get_binary_email('HeaderTests/Test Email 3.txt')
        header = EmailHeader(msg)
        header.x_spam_flag = True
        result = email_header.x_spam_flag().extract(header)
        self.assertTrue(result)

    def test_number_of_words_subject(self):
        msg = utils.get_binary_email('HeaderTests/Test Email 1.txt')
        header = EmailHeader(msg)
        header.subject = 'asdf%sl$$ sd*_sdlfj'
        result = email_header.number_of_special_characters_subject().extract(header)
        self.assertEqual(5, result)
