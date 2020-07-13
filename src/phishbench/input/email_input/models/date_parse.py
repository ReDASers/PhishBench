import calendar
import re
from datetime import datetime, timezone, timedelta

month_dict = {v: k for k, v in enumerate(calendar.month_abbr)}

DATE_REGEX = re.compile(r'(\d{1,2})\s+(\w{3})\s+(\d{4}|\d{2})')


def consume_space(time_str: str):
    match = re.match(r'\s+', time_str)
    if match:
        time_str = time_str[match.end():]
    return time_str


def parse_time(time_str: str):
    match = re.match(r'(\d{2})[\.:](\d{2})([\.:](\d{2}))?', time_str)
    if not match:
        raise ValueError('Invalid time: {}'.format(time_str))
    groups = match.groups()
    hour = int(groups[0])
    minutes = int(groups[1])
    if groups[3]:
        secs = int(groups[3])
    else:
        secs = 0
    time_str = consume_space(time_str[match.end():])
    return hour, minutes, secs, time_str


def parse_timezone(time_str: str):
    match = re.match(r'(([+-])(\d{2})(\d{2})|GMT)', time_str)
    if match:
        if match.group() == 'GMT':
            offset = timedelta()
            time_zone = timezone(offset)
        else:
            group = match.groups()
            # print(group)
            tz_hour = group[1] + group[2]
            tz_hour = int(tz_hour)
            tz_mins = group[1] + group[3]
            tz_mins = int(tz_mins)
            offset = timedelta(hours=tz_hour, minutes=tz_mins)
            time_zone = timezone(offset)
        time_str = consume_space(time_str[match.end():])
        return time_zone, time_str
    else:
        return None, time_str


def parse_date(date_str: str):
    match = DATE_REGEX.match(date_str)
    if not match:
        raise ValueError("Invalid date")
    groups = match.groups()
    day = int(groups[0])
    month = month_dict[groups[1]]
    year = groups[2]
    if len(year) == 2:
        if int(year) < 70:
            year = '20' + year
        else:
            year = '19' + year
    year = int(year)
    date_str = consume_space(date_str[match.end():])
    return day, month, year, date_str


def parse_email_datetime(date_str: str):
    """
    Parses an email date according to RFC 5322 section 3.3
    :param date_str: The date to parse
    :return: A `datetime` object containing the parsed date.
    """
    # We don't care about the day of the week.
    match = re.match(r"\w{3},\s+", date_str)
    if match:
        date_str = date_str[match.end():]

    day, month, year, date_str = parse_date(date_str)
    hour, mins, secs, date_str = parse_time(date_str)
    time_zone, date_str = parse_timezone(date_str)

    result = datetime(year=year, month=month, day=day, hour=hour, minute=mins, second=secs, tzinfo=time_zone)

    return result
