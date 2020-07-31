"""
This is an internal module which handles email headers
"""
import email
import re
from datetime import datetime
from email.header import Header as RawHeader
from email.message import Message
from typing import List, Union

import chardet

from phishbench.utils import Globals
from .date_parse import parse_email_datetime

EMAIL_ADDRESS_REGEX = re.compile(r"<.*@[a-zA-Z0-9.\-_]*", flags=re.MULTILINE | re.IGNORECASE)
EMAIL_ADDRESS_NAME_REGEX = re.compile(r'"?.*"? <?', flags=re.MULTILINE | re.IGNORECASE)
EMAIL_ADDRESS_DOMAIN_REGEX = re.compile(r"@.*", flags=re.MULTILINE | re.IGNORECASE)

EMAIL_DATE_FORMAT_REGEX = re.compile(r'((?P<DayName>\w{3}),\s+)?' +
                                     r'(?P<Day>\d{1,2})\s+' +
                                     r'(?P<Month>\w+)\s+' +
                                     r'(?P<Year>\d{4})\s+' +
                                     r'(?P<time>\d{2}:\d{2}(:\d{2})?)' +
                                     r'(\s+(?P<zone>[\+-]\d{4}))?', flags=re.MULTILINE)


def parse_address_list(raw: str) -> List[str]:
    """
    Parses an address-list according to the rfc2822 section 3.4
    Parameters
    ----------
    raw: str
        The raw address-list
    Returns
    -------
        A list of addresses
    """
    if not raw:
        return []
    raw = re.sub(r'\s+', ' ', raw)
    return [x.strip() for x in raw.split(',')]


def decode_header_field(msg: Message, field: str) -> Union[str, None]:
    """
    Decodes a header field from a msg object
    :param msg: The Message object to process
    :param field: The field to decode
    :return:
    """
    # Recipient
    if msg[field] is not None:
        header = msg[field]
        if isinstance(header, str):
            # Already parsed
            return header
        if isinstance(header, RawHeader):
            binary = email.header.decode_header(header)[0][0]
            encoding = chardet.detect(binary)['encoding']
            return binary.decode(encoding)
    return None


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

    # pylint: disable=too-few-public-methods
    # pylint: disable=too-many-instance-attributes
    # Adapted from original PhishBench Header extraction code
    def __init__(self, msg: Message):
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        # Get the raw headers
        self.header = ''
        for k, i in msg.items():
            self.header = self.header + str(k) + ": " + str(i) + "\n"

        self.__parse_date(msg)

        self.__parse_subject(msg)

        self.__parse_reply_to(msg)

        self.__parse_sender(msg)

        self.__parse_to(msg)

        self.__parse_recipient(msg)

        self.__parse_cc(msg)

        # Message-Id
        if msg['Message-ID']:
            self.message_id = decode_header_field(msg, 'Message-ID').strip('>').strip('<')
        else:
            self.message_id = None

        self.__parse_x_headers(msg)

        # Authentication-Results
        if msg["Authentication-Results"]:
            self.authentication_results = msg["Authentication-Results"]
        else:
            self.authentication_results = []

        # Received
        if msg["Received"]:
            self.received = msg.get_all("Received")
            # print("received: {}".format(received))
        else:
            self.received = []

        self.__parse_mime(msg)

        self.__parse_flags(msg)

    def __parse_x_headers(self, msg):
        # X-Priority
        try:
            if msg['X-Priority']:
                self.x_priority = int(msg['X-Priority'])
            else:
                self.x_priority = None
        except ValueError as exception:
            Globals.logger.debug("Exception: %s handled", exception)
            self.x_priority = None

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

    def __parse_mime(self, msg):
        # MIME version
        if msg['MIME-Version']:
            self.mime_version = msg['MIME-Version']
        else:
            self.mime_version = None

    def __parse_subject(self, msg):
        # Subject
        if msg['Subject'] is not None:
            self.subject = decode_header_field(msg, 'Subject')
        else:
            self.subject = None

    def __parse_to(self, msg: Message):
        if not msg['To']:
            self.to = []
        to = decode_header_field(msg, 'To')
        if to:
            self.to = parse_address_list(to)
        else:
            self.to = []

    def __parse_flags(self, msg: Message):
        # X-Original-Authentication-Results
        if msg["X-Original-Authentication-Results"]:
            self.x_original_authentication_results = True
        else:
            self.x_original_authentication_results = False

        # X-Spam_flag
        if msg["X-Spam_flag"]:
            self.x_spam_flag = True
        else:
            self.x_spam_flag = False

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

    def __parse_date(self, msg):
        # orig-date
        try:
            if msg['Date']:
                self.orig_date: datetime = parse_email_datetime(msg['Date'])
            else:
                self.orig_date = None
        except ValueError as exception:
            Globals.logger.debug("Exception: %s handled", exception)
            self.orig_date = None

    def __parse_sender(self, msg):
        # Sender
        if msg['Sender']:
            self.sender_full = decode_header_field(msg, 'Sender')
        else:
            # By the Email specification, either the Sender field or From field must be included. If it isn't then
            # msg['From'] simply returns None
            self.sender_full = decode_header_field(msg, 'From')

        if self.sender_full and re.findall(EMAIL_ADDRESS_NAME_REGEX, self.sender_full):
            self.sender_name = re.findall(EMAIL_ADDRESS_NAME_REGEX, self.sender_full)[0] \
                .strip('"').strip(' <').strip('"')
        else:
            self.sender_name = None

        if self.sender_full and re.findall(EMAIL_ADDRESS_REGEX, self.sender_full):
            self.sender_email_address = re.findall(EMAIL_ADDRESS_REGEX, self.sender_full)[0] \
                .strip("<").strip(">")
        else:
            self.sender_email_address = None

    def __parse_recipient(self, msg):
        if msg['Delivered-To']:
            self.recipient_full = decode_header_field(msg, 'Delivered-To')
        elif msg['X-Envelope-To']:
            self.recipient_full = decode_header_field(msg, 'X-Envelope-To')
        elif len(self.to) > 0:
            self.recipient_full = self.to[0]
        else:
            self.recipient_full = None

        if self.recipient_full and re.findall(EMAIL_ADDRESS_NAME_REGEX, self.recipient_full):
            self.recipient_name = re.findall(EMAIL_ADDRESS_NAME_REGEX, self.recipient_full)[0] \
                .strip('"').strip(' <').strip('"')
        else:
            self.recipient_name = None

        if self.recipient_full and re.findall(EMAIL_ADDRESS_REGEX, self.recipient_full):
            self.recipient_email_address = re.findall(EMAIL_ADDRESS_REGEX, self.recipient_full)[0] \
                .strip('<').strip('>')
        else:
            self.recipient_email_address = None

    def __parse_cc(self, msg):
        # CC
        if msg["Cc"]:
            cc = decode_header_field(msg, 'Cc')
            if cc:
                self.cc = parse_address_list(cc)
            else:
                self.cc = []
        else:
            self.cc = []

        # BCC
        if msg["Bcc"]:
            cc = decode_header_field(msg, 'Bcc')
            if cc:
                self.bcc = parse_address_list(cc)
            else:
                self.bcc = []
        else:
            self.bcc = []

    def __parse_reply_to(self, msg):
        # Return-Path
        if msg['Return-Path']:
            self.return_path = decode_header_field(msg, 'Return-Path').strip('>').strip('<')
        else:
            self.return_path = None

        # Reply To
        if msg['Reply-To'] is not None:
            rt = decode_header_field(msg, 'Reply-To')
            if rt:
                self.reply_to = parse_address_list(rt)
            else:
                self.reply_to = []
        else:
            self.reply_to = []
