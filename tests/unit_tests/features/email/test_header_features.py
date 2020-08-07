import unittest

from phishbench.feature_extraction.email.features import email_header
from phishbench.input.email_input.models import EmailHeader
from ....sample_emails import utils


class TestHeaderFeatures(unittest.TestCase):

    def test_number_of_words_subject(self):
        msg = utils.get_binary_email('HeaderTests/Test Email 1.txt')
        header = EmailHeader(msg)
        header.subject = 'asdf%sl$$ sd*_sdlfj'
        result = email_header.number_of_special_characters_subject(header)
        self.assertEqual(result, 5)
