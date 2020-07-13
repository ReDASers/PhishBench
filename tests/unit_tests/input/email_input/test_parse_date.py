import unittest

from phishbench.input.email_input.models.date_parse import parse_date, parse_time, parse_email_datetime


class TestParseDate(unittest.TestCase):

    def test_parse_time(self):
        time = "03:14:07 +0500"
        result = parse_time(time)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 3)
        self.assertEqual(result[1], 14)
        self.assertEqual(result[2], 7)
        self.assertEqual(result[3], '+0500')

    def test_parse_time_no_secs(self):
        time = "03:14 +0500"
        result = parse_time(time)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 3)
        self.assertEqual(result[1], 14)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], '+0500')

    def test_parse_email_date_bad(self):
        raw = "BACON!"

        self.assertRaises(ValueError, parse_date, raw)

    def test_parse_date(self):
        raw = '14 Apr 2015 03:14 +0500'
        result = parse_date(raw)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 14)
        self.assertEqual(result[1], 4)
        self.assertEqual(result[2], 2015)
        self.assertEqual(result[3], '03:14 +0500')

    def test_parse_date_2d(self):
        raw = '14 Apr 15 03:14 +0500'
        result = parse_date(raw)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 14)
        self.assertEqual(result[1], 4)
        self.assertEqual(result[2], 2015)
        self.assertEqual(result[3], '03:14 +0500')

    def test_parse_email_date(self):
        raw = 'Mon, 14 Apr 2015 16:08:50 +0500'
        date = parse_email_datetime(raw)

        self.assertEqual(2015, date.year)
        self.assertEqual(14, date.day)
        self.assertEqual(4, date.month)
        self.assertEqual(16, date.hour)
        self.assertEqual(8, date.minute)
        self.assertEqual(50, date.second)
        self.assertEqual("UTC+05:00", str(date.tzinfo))

    def test_parse_email_date_no_zone(self):
        raw = 'Mon, 14 Apr 2015 16:08:50'

        date = parse_email_datetime(raw)

        self.assertEqual(2015, date.year)
        self.assertEqual(14, date.day)
        self.assertEqual(4, date.month)
        self.assertEqual(16, date.hour)
        self.assertEqual(8, date.minute)
        self.assertEqual(50, date.second)

    def test_parse_email_date_no_name(self):
        raw = '14 Apr 2015 16:08:50'

        date = parse_email_datetime(raw)

        self.assertEqual(2015, date.year)
        self.assertEqual(14, date.day)
        self.assertEqual(4, date.month)
        self.assertEqual(16, date.hour)
        self.assertEqual(8, date.minute)
        self.assertEqual(50, date.second)

    def test_parse_email_date_short_time(self):
        raw = 'Mon, 14 Apr 2015 16:08'

        date = parse_email_datetime(raw)

        self.assertEqual(2015, date.year)
        self.assertEqual(14, date.day)
        self.assertEqual(4, date.month)
        self.assertEqual(16, date.hour)
        self.assertEqual(8, date.minute)
        self.assertEqual(0, date.second)

    def test_parse_invalid_date(self):
        raw = 'Mon, 29 Feb 2015 16:08'
        self.assertRaises(ValueError, parse_email_datetime, raw)
