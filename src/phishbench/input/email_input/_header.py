import email
import re
from email.message import Message
from typing import List
import datetime

from ...utils import Globals

email_address_regex = re.compile(r"<.*@[a-zA-Z0-9.\-_]*", flags=re.MULTILINE | re.IGNORECASE)
email_address_name_regex = re.compile(r'"?.*"? <?', flags=re.MULTILINE | re.IGNORECASE)
email_address_domain_regex = re.compile(r"@.*", flags=re.MULTILINE | re.IGNORECASE)

email_date_format_regex = re.compile(r'((?P<DayName>\w{3}),\s+)?' +
                                     r'(?P<Day>\d{1,2})\s+' +
                                     r'(?P<Month>\w+)\s+' +
                                     r'(?P<Year>\d{4})\s+' +
                                     r'(?P<time>\d{2}:\d{2}(:\d{2})?)' +
                                     r'(\s+(?P<zone>[\+-]\d{4}))?', flags=re.MULTILINE)


def parse_address_list(raw: str) -> List[str]:
    if not raw:
        return []
    raw = re.sub(r'\s+', ' ', raw)
    return [x.strip() for x in raw.split(',')]


def parse_email_date(raw: str):
    # Parse date format from rfc2822 section-3.3
    match = email_date_format_regex.match(raw)
    year = int(match.group('Year'))
    month = match.group('Month')
    day = int(match.group('Day'))
    time = match.group('time')
    if len(time) == 5:
        time = time + ":00"
    date_string = "%4d %s %2d %s" % (year, month, day, time)
    zone = match.group('zone')
    if zone:
        date_string = date_string + " " + zone
        format_string = '%Y %b %d %H:%M:%S %z'
    else:
        format_string = '%Y %b %d %H:%M:%S'
    return datetime.datetime.strptime(date_string, format_string)


class EmailHeader:
    """
    Represents the header of an email

    Attributes
    ----------
    orig_date : datetime
        The origination date of the email
    x_priority : int
        The X-Priority header value. If present, an integer between 1 and 5. Otherwise, None
    subject : str
        The value of the subject header field if present. None otherwise.
    return_path : str
        The return path of the email without angle brackets
    reply_to : List[str]
        The reply-to values
    sender_full : str
        The sender of the email.
    sender_name : str
        The sender's display name
    sender_email_address : str
        The email address of the sender
    to : List[str]
        The raw mailboxes in the To: field
    recipient_full : str
        The mailbox the email was sent to, or the first mailbox in the To field
        if we cannot figure out who received the email
    recipient_name : str
        The name of the recipient
    recipient_email_address : str
        The email address of the recipient
    message_id : str
        A unique message identifier for the email.
    x_mailer : str
        The desktop client which sent the email, as indicated by the X-Mailer header.
    x_originating_hostname : str
        The originating hostname if available
    x_originating_ip : str
        The originating ip if available
    x_virus_scanned : bool
        Whether or not the email has been scanned for a virus
    dkim_signed : bool
        Whether or no the email has a DKIM signature
    received_spf : bool
        Whether or not the Received-SPF header is present in the email
    x_original_authentication_results : bool
        Whether or not the X-Original-Authentication-Results header is present in the email
    authentication_results : str
        The contents of the Authentication-Results header
    received : List[str]
        A list containing the Received headers of the email
    mime_version : str
        The value of the MIME-Version header field
    """

    # Adapted from original PhishBench Header extraction code
    def __init__(self, msg: Message):
        parser = email.parser.HeaderParser()
        msg = parser.parsestr(msg.as_string())
        # Get the raw headers
        self.header = ''
        for k, i in msg.items():
            self.header = self.header + str(k) + ": " + str(i) + "\n"

        # orig-date
        try:
            if msg['Date']:
                self.orig_date = parse_email_date(msg['Date'])
            else:
                self.orig_date = None
        except Exception as exception:
            Globals.logger.debug("exception: " + str(exception))
            Globals.logger.debug("Exception handled")
            self.orig_date = None

        # X-Priority
        try:
            if msg['X-Priority']:
                self.x_priority = int(msg['X-Priority'])
            else:
                self.x_priority = None
        except ValueError as exception:
            Globals.logger.debug("exception: " + str(exception))
            Globals.logger.debug("Exception handled")
            self.x_priority = None

        # Subject
        if msg['Subject'] is not None:
            self.subject = msg['Subject']
        else:
            self.subject = None

        # Return-Path
        if msg['Return-Path']:
            self.return_path = msg['Return-Path'].strip('>').strip('<')
        else:
            self.return_path = None

        # Reply To
        if msg['Reply-To'] is not None:
            self.reply_to = parse_address_list(msg['Reply-To'])
        else:
            self.reply_to = []

        # Sender
        try:
            if msg['Sender']:
                self.sender_full = msg['Sender']
            else:
                # By the Email specification, either the Sender field or From field must be included
                self.sender_full = msg['From']
            if re.findall(email_address_name_regex, self.sender_full):
                self.sender_name = re.findall(email_address_name_regex, self.sender_full)[0] \
                    .strip('"').strip(' <').strip('"')
            else:
                self.sender_name = None
        except Exception as exception:
            Globals.logger.debug("exception: " + str(exception))
            Globals.logger.debug("Exception handled")
            self.sender_name = None
        try:
            if re.findall(email_address_regex, self.sender_full):
                self.sender_email_address = re.findall(email_address_regex, self.sender_full)[0] \
                    .strip("<").strip(">")
            else:
                self.sender_email_address = None
        except Exception as exception:
            Globals.logger.debug("exception: " + str(exception))
            Globals.logger.debug("Exception handled")
            self.sender_email_address = None

        # Recipient
        if msg['To']:
            self.to = parse_address_list(msg['To'])
        if msg['Delivered-To']:
            self.recipient_full = msg['Delivered-To']
        elif msg['X-Envelope-To']:
            self.recipient_full = msg['X-Envelope-To']
        elif len(self.to) > 0:
            self.recipient_full = self.to[0]
        else:
            self.recipient_full = None

        try:
            if self.recipient_full and re.findall(email_address_name_regex, self.recipient_full):
                self.recipient_name = re.findall(email_address_name_regex, self.recipient_full)[0] \
                    .strip('"').strip(' <').strip('"')
            else:
                self.recipient_name = None
        except Exception as exception:
            Globals.logger.debug("exception: " + str(exception))
            Globals.logger.debug("Exception handled")
            self.recipient_name = None

        try:
            match = re.findall(email_address_regex, self.recipient_full)
            if match:
                self.recipient_email_address = match[0].strip('<').strip('>')
            else:
                self.recipient_email_address = None
        except Exception as exception:
            Globals.logger.debug("exception: " + str(exception))
            Globals.logger.debug("Exception handled")
            self.recipient_email_address = None

        # CC
        if msg["Cc"]:
            self.cc = parse_address_list(msg['Cc'])
        else:
            self.cc = []

        # Bcc
        if msg["Bcc"]:
            self.bcc = parse_address_list(msg["Bcc"])
        else:
            self.bcc = []

        # Message-Id
        if msg['Message-ID']:
            self.message_id = msg['Message-ID'].strip('>').strip('<')
        else:
            self.message_id = None

        # X-mailer
        if msg['X-mailer']:
            self.x_mailer = msg['X-mailer']
        else:
            self.x_mailer = None

        # X-originating-hostname
        if msg["X-originating-hostname"]:
            self.x_originating_hostname = msg["X-originating-hostname"]
        else:
            self.x_originating_hostname = None

        # X-originating-ip
        if msg["X-originating-ip"]:
            self.x_originating_ip = msg["X-originating-ip"]
        else:
            self.x_originating_ip = None

        # Virus Scanned
        if msg["X-virus-scanned"]:
            self.x_virus_scanned = True
        else:
            self.x_virus_scanned = False

        # DKIM-Signature
        if msg["DKIM-Signature"]:
            self.dkim_signed = True
        else:
            self.dkim_signed = False

        # Received-SPF
        if msg["Received-SPF"]:
            # received_spf=msg["Received-SPF"]
            self.received_spf = True
        else:
            # received_spf="None"
            self.received_spf = False

        #X-Original-Authentication-Results
        if msg["X-Original-Authentication-Results"]:
            # x_original_authentication_results = msg["X-Original-Authentication-Results"]
            self.x_original_authentication_results = True
        else:
            self.x_original_authentication_results = False

        # Authentication-Results
        if msg["Authentication-Results"]:
            self.authentication_results = msg["Authentication-Results"]
        else:
            self.authentication_results = None

        # Received
        if msg["Received"]:
            self.received = msg.get_all("Received")
            # print("received: {}".format(received))
        else:
            self.received = []
        #
        if msg['MIME-Version']:
            self.mime_version = msg['MIME-Version']
        else:
            self.mime_version = None