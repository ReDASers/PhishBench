import re
from email.message import Message
from io import StringIO

import chardet
import lxml
from bs4 import BeautifulSoup
from lxml.html.clean import Cleaner

HEX_REGEX = re.compile(r"0x[0-9a-f]*?,?", flags=re.IGNORECASE | re.MULTILINE)
UNDERSCORE_REGEX = re.compile(r"_+", flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)


def get_charset(part):
    """
    Retrieves the declared charset of the part
    :param part: The part to get the charset from
    :return:
    The charset used to encode the part, or None if it is not found
    """
    if part.get_charset() is not None:
        return part.get_charset()
    content_type_string = part['Content-Type']
    if not content_type_string:
        return None
    if 'charset=' in content_type_string:
        charset = content_type_string.split('charset=')[1]
        match = re.match(r'"(.+?)"', charset)
        if match:
            return match.groups()[0]
        # No quotes
        return charset.split()[0]
    return None


def decode_text_part(part):
    payload = part.get_payload(decode=True)
    if len(payload) == 0:
        return None, None
    charset = get_charset(part)
    if charset is not None:
        # self.charset_list.append(charset)
        if isinstance(payload, str):
            # Payload has already been decoded
            payload = payload.strip()
        else:
            try:
                payload = payload.decode(charset).strip()
            except UnicodeError:
                raw_data = part.get_payload()
                if isinstance(raw_data, str):
                    payload = raw_data
                else:
                    return None, None
    else:
        encoding = chardet.detect(payload)['encoding']
        if encoding:
            charset = encoding.lower()
            try:
                payload = payload.decode(encoding=charset).strip()
            except UnicodeError:
                return None, None
    return payload, charset


class EmailBody:
    """
    A class representing the body of an email.
    Attributes
    ----------
    text : str
        The raw text of the email.
    html : str
        The cleaned html of the email.
    is_html : bool
        Whether or not the email is html
    num_attachment: int
        The number of attachments the email contains
    content_disposition_list: List[str]
        A list of the content dispositions of each part of the email
    content_type_list : List[str]
        A list of the content types of each part of the email
    content_transfer_encoding_list: List[str]
        A list of the content transfer encodings for each part
    file_extension_list: List[str]
        A list containing the file extensions for each attachment
    charset_list:
        A list containing the charsets for each part
    """

    def __init__(self, msg: Message):
        self.text = None
        self.html = None
        self.is_html = False
        self.content_transfer_encoding_list = []
        self.charset_list = []
        self.num_attachment = 0
        self.content_type_list = []
        self.content_disposition_list = []
        self.file_extension_list = []
        self.defects = []
        self.raw_msg = msg
        self.__parse_msg(msg)

    def __parse_msg(self, msg: Message):
        for part in msg.walk():
            if part.is_multipart():
                # We're only interested in the leaf nodes of the email tree
                continue

            content_type = part.get_content_type()
            self.content_type_list.append(content_type)

            content_disposition = part.get_content_disposition()
            self.content_disposition_list.append(content_disposition)

            self.content_transfer_encoding_list.append(part.get('Content-Transfer-Encoding'))

            if content_disposition == 'attachment':
                self.num_attachment += 1

            if part.get_filename():
                if '.' in part.get_filename():
                    _, ext = part.get_filename().rsplit('.', maxsplit=1)
                    self.file_extension_list.append(ext)
                else:
                    self.file_extension_list.append(None)

            if content_type == 'text/plain':
                self.__parse_text_part(part)
            elif content_type == 'text/html':
                self.is_html = True
                self.__parse_html_part(part)

    def __parse_text_part(self, part):
        payload, charset = decode_text_part(part)

        if charset:
            self.charset_list.append(charset)

        if self.text and payload:
            self.text += '\n'
            self.text += payload
        elif payload:
            self.text = payload

    def __parse_html_part(self, part):
        payload, charset = decode_text_part(part)
        if charset:
            self.charset_list.append(charset)

        cleaner = Cleaner()
        cleaner.javascript = True
        cleaner.style = True
        self.html = lxml.html.tostring(cleaner.clean_html(lxml.html.parse(StringIO(payload))))
        if not self.text:
            soup = BeautifulSoup(self.html, 'html.parser')
            self.text = soup.get_text()
