import email
import unittest
from os import path
from unittest.mock import patch, MagicMock

from phishbench.input.email_input import EmailHeader, EmailBody
from phishbench.input.email_input._header import parse_address_list
from phishbench.input.email_input._header import parse_email_date


def get_email(filename):
    current_loc = path.dirname(path.abspath(__file__))
    loc = path.join(current_loc, filename)
    with open(loc, 'r') as file:
        return email.message_from_file(file)


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

    def test_parse_email_date_bad(self):
        raw = "BACON!"

        self.assertRaises(ValueError, parse_email_date, raw)


    def test_parse_email_date(self):
        raw = 'Mon, 14 Apr 2015 16:08:50 +0500'
        date = parse_email_date(raw)

        self.assertEqual(2015, date.year)
        self.assertEqual(14, date.day)
        self.assertEqual(4, date.month)
        self.assertEqual(16, date.hour)
        self.assertEqual(8, date.minute)
        self.assertEqual(50, date.second)
        self.assertEqual("UTC+05:00", str(date.tzinfo))

    def test_parse_email_date_no_zone(self):
        raw = 'Mon, 14 Apr 2015 16:08:50'

        date = parse_email_date(raw)

        self.assertEqual(2015, date.year)
        self.assertEqual(14, date.day)
        self.assertEqual(4, date.month)
        self.assertEqual(16, date.hour)
        self.assertEqual(8, date.minute)
        self.assertEqual(50, date.second)

    def test_parse_email_date_no_name(self):
        raw = '14 Apr 2015 16:08:50'

        date = parse_email_date(raw)

        self.assertEqual(2015, date.year)
        self.assertEqual(14, date.day)
        self.assertEqual(4, date.month)
        self.assertEqual(16, date.hour)
        self.assertEqual(8, date.minute)
        self.assertEqual(50, date.second)

    def test_parse_email_date_short_time(self):
        raw = 'Mon, 14 Apr 2015 16:08'

        date = parse_email_date(raw)

        self.assertEqual(2015, date.year)
        self.assertEqual(14, date.day)
        self.assertEqual(4, date.month)
        self.assertEqual(16, date.hour)
        self.assertEqual(8, date.minute)
        self.assertEqual(0, date.second)

    def test_date(self):
        msg = get_email("Resources/Test Email 1.txt")
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
        msg = get_email("Resources/Test Email 2.txt")

        header = EmailHeader(msg)

        self.assertEqual(1, header.x_priority)

    @patch('phishbench.utils.Globals.logger.debug')
    def test_X_priority_error(self, l_mock):
        msg = get_email("Resources/Test Email 2.txt")
        del msg['X-Priority']
        msg['X-Priority'] = "BAD"

        header = EmailHeader(msg)

        self.assertIsNone(header.x_priority)
        l_mock.assert_called()

    def test_X_priority_null(self):
        msg = get_email("Resources/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertIsNone(header.x_priority)

    def test_subject(self):
        msg = get_email("Resources/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual('Re: Is there a reason', header.subject)

    def test_subject_empty(self):
        msg = get_email("Resources/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertEqual("", header.subject)

    def test_subject_null(self):
        msg = get_email("Resources/Test Email 3.txt")
        header = EmailHeader(msg)
        self.assertIsNone(header.subject)

    def test_return_path(self):
        msg = get_email("Resources/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual("user@domain.com", header.return_path)

    def test_return_path_bracket(self):
        msg = get_email("Resources/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertEqual("user@domain.com", header.return_path)

    def test_return_path_null(self):
        msg = get_email("Resources/Test Email 3.txt")
        header = EmailHeader(msg)
        self.assertIsNone(header.return_path)

    def test_reply_to_empty(self):
        msg = get_email("Resources/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual(0, len(header.reply_to))

    @patch('phishbench.input.email_input._header.parse_address_list')
    def test_reply_to(self, m_mock: MagicMock):
        msg = get_email("Resources/Test Email 2.txt")

        EmailHeader(msg)
        m_mock.assert_any_call("\"User\" <user@domain.com>, Bob <user2@domain.com>")

    def test_sender(self):
        msg = get_email("Resources/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual("Matthew Budman <matthewb@annapurnapics.com>", header.sender_full)
        self.assertEqual("Matthew Budman", header.sender_name)
        self.assertEqual("matthewb@annapurnapics.com", header.sender_email_address)

    @patch('phishbench.input.email_input._header.parse_address_list')
    def test_to(self, a_mock):
        msg = get_email("Resources/Test Email 2.txt")
        EmailHeader(msg)
        a_mock.assert_any_call("\"User\" <user@domain.com>, Bob <user2@domain.com>")

    def test_recipient(self):
        msg = get_email("Resources/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual("\"User\" <user@domain.com>", header.recipient_full)
        self.assertEqual("User", header.recipient_name)
        self.assertEqual("user@domain.com", header.recipient_email_address)

    def test_cc(self):
        msg = get_email("Resources/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual(5, len(header.cc))
        self.assertEqual('Charles Roven <user@domain.com>', header.cc[0])
        self.assertEqual('Amy pascal <user@domain.com>', header.cc[1])
        self.assertEqual('Doug Belgrad <user@domain.com>', header.cc[2])
        self.assertEqual('Andre Caraco <user@domain.com>', header.cc[3])
        self.assertEqual('Ekta Farrar <user@domain.com>', header.cc[4])

    def test_cc_empty(self):
        msg = get_email("Resources/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertEqual(0, len(header.cc))

    def test_cc_null(self):
        msg = get_email("Resources/Test Email 3.txt")
        header = EmailHeader(msg)
        self.assertEqual(0, len(header.cc))

    @patch('phishbench.input.email_input._header.parse_address_list')
    def test_bcc(self, m_mock):
        msg = get_email("Resources/Test Email 2.txt")
        EmailHeader(msg)
        m_mock.assert_any_call("Anna <anna@domain.com>, Joe <user2@domain.com>,\n Sue <user3@domain.com>")

    def test_message_id(self):
        msg = get_email("Resources/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual("asldfjalsdjf@domain.com", header.message_id)

    def test_x_mailer_none(self):
        msg = get_email("Resources/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertIsNone(header.x_mailer)

    def test_x_mailer(self):
        msg = get_email("Resources/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertEqual("Microsoft Outlook 16.0", header.x_mailer)

    def test_dkim_signed(self):
        msg = get_email("Resources/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertTrue(header.dkim_signed)

    def test_dkim_unsigned(self):
        msg = get_email("Resources/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertFalse(header.dkim_signed)

    def test_received_count(self):
        msg = get_email("Resources/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertEqual(8, len(header.received))

    def test_mime_version(self):
        msg = get_email("Resources/Test Email 1.txt")
        header = EmailHeader(msg)
        self.assertEqual('1.0', header.mime_version)

    def test_mime_version_none(self):
        msg = get_email("Resources/Test Email 2.txt")
        header = EmailHeader(msg)
        self.assertIsNone(header.mime_version)


class TestEmailBody(unittest.TestCase):

    def test_email_body(self):
        msg = get_email("Resources/Test Email 1.txt")
        body = EmailBody(msg)
        expected = 'We didn\'t win anything... ' \
                   'Jennifer won best actress for hunger games and hunger games won best film. ' \
                   'But she was not there. \n' \
                   'Matthew Budman\n\nAnnapurna Pictures\n\n310-724-5678\n\nsent from my iPhone.\n\n'
        self.assertEqual(expected.strip(), body.text.strip())
