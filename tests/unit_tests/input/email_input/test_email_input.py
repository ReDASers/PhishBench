import unittest

from phishbench.input.email_input import read_email_from_file
from ....sample_emails import utils


class TestRawInput(unittest.TestCase):

    def test_read_email(self):
        # Testcase for Issue #213
        filepath = utils.get_relative_path('BodyTests/phishing3_1113.txt')
        msg = read_email_from_file(filepath)
        for header in msg:
            self.assertIsInstance(msg[header], str)
