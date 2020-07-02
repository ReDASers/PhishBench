import unittest

from phishbench.input.email_input.models import EmailBody
from .utils import get_email, get_relative_path


class TestEmailBody(unittest.TestCase):

    def test_email_body(self):
        msg = get_email("Resources/BodyTests/Test Body Email 1.txt")
        body = EmailBody(msg)
        with open(get_relative_path('Resources/BodyTests/test_body_1.txt')) as f:
            expected = f.read().strip()
        self.maxDiff = None
        self.assertEqual(expected, body.text.strip())

    def test_email_body2(self):
        msg = get_email("Resources/BodyTests/test body email 2.txt")
        body = EmailBody(msg)

        with open(get_relative_path('Resources/BodyTests/test_body_2.txt')) as f:
            expected = f.read()

        self.maxDiff = None
        print(body.text.strip())
        self.assertEqual(expected, body.text.strip())

    def test_email_body3(self):
        msg = get_email("Resources/BodyTests/Test Body Email 3.txt")
        body = EmailBody(msg)
        with open(get_relative_path('Resources/BodyTests/test_body_3.txt')) as f:
            expected = f.read().strip()

        self.maxDiff = None
        self.assertEqual(expected, body.text.strip())

    def test_email_html(self):
        msg = get_email("Resources/BodyTests/test body email 2.txt")
        body = EmailBody(msg)
        print(body.html)

    def test_email_body_attachment(self):
        msg = get_email("Resources/BodyTests/test body email 2.txt")
        body = EmailBody(msg)

        self.assertTrue(body.is_html)
        self.assertEqual(1, body.num_attachment)
        self.assertIn('docx', body.file_extension_list)
