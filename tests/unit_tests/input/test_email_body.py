import unittest

from phishbench.input.email_input.models import EmailBody
from .utils import get_email, get_relative_path


class TestEmailBody(unittest.TestCase):

    def test_email_body(self):
        msg = get_email("Resources/BodyTests/Test Body Email 1.txt")
        body = EmailBody(msg)
        with open(get_relative_path('Resources/BodyTests/test_body_1.txt')) as f:
            expected = f.read().strip()

        self.assertEqual(expected, body.text.strip())
