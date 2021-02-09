# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=import-error

import unittest

from phishbench.input.email_input.models._body import clean_html
from phishbench.input.email_input.models import EmailBody
import tests.sample_emails.utils as utils


class TestEmailBody(unittest.TestCase):

    def test_email_body(self):
        msg = utils.get_binary_email("BodyTests/Test Body Email 1.txt")
        body = EmailBody(msg)
        with open(utils.get_relative_path('BodyTests/test_body_1.txt')) as f:
            expected = f.read().strip()
        # pylint: disable=invalid-name
        self.maxDiff = None
        self.assertEqual(expected, body.text.strip())
        self.assertIn('us-ascii', body.charset_list)

    def test_email_body2(self):
        msg = utils.get_binary_email("BodyTests/test body email 2.txt")
        body = EmailBody(msg)

        with open(utils.get_relative_path('BodyTests/test_body_2.txt')) as f:
            expected = f.read()

        # pylint: disable=invalid-name
        self.maxDiff = None
        self.assertEqual(expected, body.text.strip())
        self.assertIn('us-ascii', body.charset_list)

    def test_email_body3(self):
        msg = utils.get_binary_email("BodyTests/Test Body Email 3.txt")
        body = EmailBody(msg)
        with open(utils.get_relative_path('BodyTests/test_body_3.txt')) as f:
            expected = f.read().strip()

        # pylint: disable=invalid-name
        self.maxDiff = None
        self.assertEqual(expected, body.text.strip())
        self.assertIn('iso-8859-1', body.charset_list)

    def test_email_body4(self):
        msg = utils.get_email_text("BodyTests/Test Body Email 4.txt")
        body = EmailBody(msg)

        with open(utils.get_relative_path('BodyTests/test_body_4.txt'), encoding='iso-8859-1') as f:
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
        msg = utils.get_binary_email("BodyTests/Test Body Email 5.txt")
        body = EmailBody(msg)
        with open(utils.get_relative_path('BodyTests/test_body_5.txt')) as f:
            expected = f.read().strip()

        self.assertEqual(expected, body.text.strip())

    def test_clean_html(self):
        self.maxDiff = None
        with open(utils.get_relative_path('BodyTests/html_dirty.html')) as f:
            html = f.read()
        cleaned = clean_html(html)

        with open(utils.get_relative_path('BodyTests/html_clean.html')) as f:
            expected = f.read()
        self.assertEqual(cleaned, expected)

    def test_email_html(self):
        msg = utils.get_binary_email("BodyTests/test body email 2.txt")
        body = EmailBody(msg)
        with open(utils.get_relative_path('BodyTests/test_body2.html')) as f:
            expected_raw = f.read()

        self.assertEqual(body.raw_html, expected_raw)
        with open(utils.get_relative_path('BodyTests/test_body2_clean.html')) as f:
            expected = f.read()
        self.assertEqual(body.html, expected)

    def test_empty_html(self):
        msg = utils.get_binary_email('BodyTests/empty_html_part.txt')
        body = EmailBody(msg)

        self.assertIsNone(body.html)
        self.assertIsNone(body.raw_html)
        self.assertFalse(body.is_html)

    def test_email_body_attachment(self):
        msg = utils.get_binary_email("BodyTests/test body email 2.txt")
        body = EmailBody(msg)

        self.assertTrue(body.is_html)
        self.assertEqual(1, body.num_attachment)
        self.assertIn('docx', body.file_extension_list)

    def test_attachment_no_filename(self):
        msg = utils.get_binary_email('BodyTests/no_filename.txt')
        body = EmailBody(msg)

        self.assertEqual(1, body.num_attachment)
        self.assertEqual(1, len(body.file_extension_list))
        self.assertIn(None, body.file_extension_list)

    def test_charset_no_quotes(self):
        msg = utils.get_binary_email("BodyTests/Test Body Email 6.txt")
        body = EmailBody(msg)

        self.assertIn('us-ascii', body.charset_list)

    def test_charset_unknown(self):
        msg = utils.get_email_text("BodyTests/Test Body Email 7.txt")
        body = EmailBody(msg)

        self.assertIn('iso-5636-2', body.charset_list)

    def test_multipart_root(self):
        msg = utils.get_email_text('BodyTests/Test_No_CT.txt')
        body = EmailBody(msg)

        expected = 'Ok. I am here til 6. Just let me know. No stress.'
        self.assertEqual(body.text.strip(), expected)

    def test_issue_291(self):
        msg = utils.get_email_text('BodyTests/Nazario_2016 (260).txt')
        body = EmailBody(msg)

        with open(utils.get_relative_path('BodyTests/Nazario_2016 (260).html'), encoding='utf-8') as f:
            expected = f.read()
        self.assertEqual(body.html, expected)
