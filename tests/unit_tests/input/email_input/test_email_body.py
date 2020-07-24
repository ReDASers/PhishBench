# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=import-error

import unittest

from phishbench.input.email_input.models import EmailBody
import tests.unit_tests.input.email_input.utils as utils


class TestEmailBody(unittest.TestCase):

    def test_email_body(self):
        msg = utils.get_binary_email("Resources/BodyTests/Test Body Email 1.txt")
        body = EmailBody(msg)
        with open(utils.get_relative_path('Resources/BodyTests/test_body_1.txt')) as f:
            expected = f.read().strip()
        # pylint: disable=invalid-name
        self.maxDiff = None
        self.assertEqual(expected, body.text.strip())
        self.assertIn('us-ascii', body.charset_list)

    def test_email_body2(self):
        msg = utils.get_binary_email("Resources/BodyTests/test body email 2.txt")
        body = EmailBody(msg)

        with open(utils.get_relative_path('Resources/BodyTests/test_body_2.txt')) as f:
            expected = f.read()

        # pylint: disable=invalid-name
        self.maxDiff = None
        self.assertEqual(expected, body.text.strip())
        self.assertIn('us-ascii', body.charset_list)

    def test_email_body3(self):
        msg = utils.get_binary_email("Resources/BodyTests/Test Body Email 3.txt")
        body = EmailBody(msg)
        with open(utils.get_relative_path('Resources/BodyTests/test_body_3.txt')) as f:
            expected = f.read().strip()

        # pylint: disable=invalid-name
        self.maxDiff = None
        self.assertEqual(expected, body.text.strip())
        self.assertIn('iso-8859-1', body.charset_list)

    def test_email_body4(self):
        msg = utils.get_email_text("Resources/BodyTests/Test Body Email 4.txt")
        body = EmailBody(msg)
        with open(utils.get_relative_path('Resources/BodyTests/test_body_4.txt')) as f:
            expected = f.read().strip()

        self.assertEqual(expected, body.text.strip())
        self.assertFalse(body.is_html)
        self.assertIsNone(body.html)
        self.assertEqual(len(body.charset_list), 1)
        self.assertIn('iso-8859-1', body.charset_list)
        self.assertEqual(len(body.file_extension_list), 0)
        self.assertIn('text/plain', body.content_type_list)
        print(body.content_transfer_encoding_list)
        self.assertEqual(len(body.content_transfer_encoding_list), 1)
        self.assertIn(None, body.content_transfer_encoding_list)
        self.assertIn(None, body.content_disposition_list)

    def test_email_body5(self):
        msg = utils.get_binary_email("Resources/BodyTests/Test Body Email 5.txt")
        body = EmailBody(msg)
        with open(utils.get_relative_path('Resources/BodyTests/test_body_5.txt')) as f:
            expected = f.read().strip()

        self.assertEqual(expected,body.text.strip())

    def test_email_html(self):
        msg = utils.get_binary_email("Resources/BodyTests/test body email 2.txt")
        body = EmailBody(msg)
        print(body.html)
        # TODO: Finish

    def test_email_body_attachment(self):
        msg = utils.get_binary_email("Resources/BodyTests/test body email 2.txt")
        body = EmailBody(msg)

        self.assertTrue(body.is_html)
        self.assertEqual(1, body.num_attachment)
        self.assertIn('docx', body.file_extension_list)

    def test_charset_no_quotes(self):
        msg = utils.get_binary_email("Resources/BodyTests/Test Body Email 6.txt")
        body = EmailBody(msg)

        self.assertIn('us-ascii', body.charset_list)
