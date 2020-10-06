import unittest
from unittest.mock import patch

from phishbench.input.email_input.models import EmailHeader
from phishbench.input.email_input.models._header import parse_address_list
from tests.sample_emails.utils import get_binary_email


class TestEmailHeader(unittest.TestCase):

    def test_parse_address_list_none(self):
        result = parse_address_list(None)
        self.assertEqual(len(result), 0)

    def test_parse_address_list_single(self):
        raw = "\"User\" <user@domain.com>"

        result = parse_address_list(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(raw, result[0])

    def test_parse_address_list_multi(self):
        raw = "Bob <bob@domain.com>, Anna <anna@domain.com>   "

        result = parse_address_list(raw)
        self.assertEqual(len(result), 2)
        self.assertEqual('Bob <bob@domain.com>', result[0])
        self.assertEqual('Anna <anna@domain.com>', result[1])

    def test_parse_address_list_multi_lines(self):
        raw = "Bob   \t<bob@domain.com>, Anna \r\n<anna@domain.com>   \n "

        result = parse_address_list(raw)
        self.assertEqual(len(result), 2)
        self.assertEqual('Bob <bob@domain.com>', result[0])
        self.assertEqual('Anna <anna@domain.com>', result[1])

    def test_date(self):
        msg = get_binary_email("HeaderTests/Test Email 1.txt")
        # Date: Mon, 14 Apr 2015 16:08:50 +0000
        header = EmailHeader(msg)

        date = header.orig_date
        self.assertEqual(2015, date.year)
        self.assertEqual(14, date.day)
        self.assertEqual(4, date.month)
        self.assertEqual(16, date.hour)
        self.assertEqual(8, date.minute)
        self.assertEqual(50, date.second)

    def test_X_priority(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")

        header = EmailHeader(msg)

        self.assertEqual(1, header.x_priority)

    @patch('phishbench.utils.phishbench_globals.logger.debug')
    def test_X_priority_error(self, l_mock):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")
        del msg['X-Priority']
        msg['X-Priority'] = "BAD"

        header = EmailHeader(msg)

        self.assertIsNone(header.x_priority)
        l_mock.assert_called()

    def test_X_priority_null(self):
        msg = get_binary_email("HeaderTests/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertIsNone(header.x_priority)

    def test_subject(self):
        msg = get_binary_email("HeaderTests/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual('Re: Is there a reason', header.subject)

    def test_subject_empty(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertEqual("", header.subject)

    def test_subject_null(self):
        msg = get_binary_email("HeaderTests/Test Email 3.txt")
        header = EmailHeader(msg)
        self.assertIsNone(header.subject)

    def test_return_path(self):
        msg = get_binary_email("HeaderTests/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual("return@domain.com", header.return_path)

    def test_return_path_bracket(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertEqual("user@domain.com", header.return_path)

    def test_return_path_null(self):
        msg = get_binary_email("HeaderTests/Test Email 3.txt")
        header = EmailHeader(msg)
        self.assertIsNone(header.return_path)

    def test_reply_to_empty(self):
        msg = get_binary_email("HeaderTests/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual(0, len(header.reply_to))

    def test_reply_to(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")

        header = EmailHeader(msg)
        self.assertEqual(len(header.reply_to), 2)
        self.assertEqual(header.reply_to[0], '"User" <user@domain.com>')
        self.assertEqual(header.reply_to[1], 'Bob <user2@domain.com>')

    def test_sender(self):
        msg = get_binary_email("HeaderTests/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual("Matthew Budman <matthewb@annapurnapics.com>", header.sender_full)
        self.assertEqual("Matthew Budman", header.sender_name)
        self.assertEqual("matthewb@annapurnapics.com", header.sender_email_address)

    def test_to(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")
        header = EmailHeader(msg)

        self.assertEqual(len(header.to), 2)
        self.assertEqual(header.to[0], '"User" <user@domain.com>')
        self.assertEqual(header.to[1], 'Bob <user2@domain.com>')

    def test_recipient(self):
        msg = get_binary_email("HeaderTests/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual("\"User\" <user@domain.com>", header.recipient_full)
        self.assertEqual("User", header.recipient_name)
        self.assertEqual("user@domain.com", header.recipient_email_address)

    def test_cc(self):
        msg = get_binary_email("HeaderTests/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual(5, len(header.cc))
        self.assertEqual('Charles Roven <user@domain.com>', header.cc[0])
        self.assertEqual('Amy pascal <user@domain.com>', header.cc[1])
        self.assertEqual('Doug Belgrad <user@domain.com>', header.cc[2])
        self.assertEqual('Andre Caraco <user@domain.com>', header.cc[3])
        self.assertEqual('Ekta Farrar <user@domain.com>', header.cc[4])

    def test_cc_empty(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertEqual(0, len(header.cc))

    def test_cc_null(self):
        msg = get_binary_email("HeaderTests/Test Email 3.txt")
        header = EmailHeader(msg)
        self.assertEqual(0, len(header.cc))

    def test_bcc(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")
        header = EmailHeader(msg)

        self.assertEqual(header.bcc[0], 'Anna <anna@domain.com>')
        self.assertEqual(header.bcc[1], 'Joe <user2@domain.com>')
        self.assertEqual(header.bcc[2], 'Sue <user3@domain.com>')

    def test_message_id(self):
        msg = get_binary_email("HeaderTests/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual("asldfjalsdjf@domain.com", header.message_id)

    def test_x_mailer_none(self):
        msg = get_binary_email("HeaderTests/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertIsNone(header.x_mailer)

    def test_x_mailer(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertEqual("Microsoft Outlook 16.0", header.x_mailer)

    def test_x_spam_flag(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")
        msg['X-Spam-Flag'] = 'YES'
        header = EmailHeader(msg)
        self.assertTrue(header.x_spam_flag)

    def test_x_spam_flag_missing(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertFalse(header.x_spam_flag)

    def test_dkim_signed(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertTrue(header.dkim_signed)

    def test_dkim_unsigned(self):
        msg = get_binary_email("HeaderTests/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertFalse(header.dkim_signed)

    def test_received_count(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertEqual(8, len(header.received))

    def test_mime_version(self):
        msg = get_binary_email("HeaderTests/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual('1.0', header.mime_version)

    def test_mime_version_none(self):
        msg = get_binary_email("HeaderTests/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertIsNone(header.mime_version)

    def test_header_windows(self):
        msg = get_binary_email("HeaderTests/Test Email 4.txt")
        header = EmailHeader(msg)
        self.assertEqual('Valérie Masson-Delmotte <Valerie.Masson@cea.fr>', header.sender_full)
        self.assertEqual('Valerie.Masson@cea.fr', header.sender_email_address)
        self.assertEqual('Valérie Masson-Delmotte', header.sender_name)
        self.assertIn('Keith Briffa <k.briffa@uea.ac.uk>', header.to)
        self.assertIn('Jonathan Overpeck <jto@u.arizona.edu>', header.to)
        self.assertIn('Eystein Jansen <eystein.jansen@geo.uib.no>', header.to)
        self.assertEqual('Re: IPCC ch9 for information and check.', header.subject)
        self.assertEqual(1, len(header.reply_to))
        self.assertIn('Valerie.Masson@cea.fr', header.reply_to)

    def test_header_unusual_encoding(self):
        """
        Tests for Issue #141
        :return:
        """
        msg = get_binary_email("HeaderTests/Test Email 5.txt")
        header = EmailHeader(msg)
        expected = "'|öS†'|öS†'|öS†'|öS†'|öS†'|öS†'|öS†'|öS†'|öS†'|öS†'|öS†'|öS†'|" \
                   "öS†'|öS†'|öS†'|öS†'|öS†'|öS†'|öS†'|öS†'|öS†'|öS†'|öS†@®àblã~Ç-Ë–"
        self.assertEqual(header.message_id, expected)
