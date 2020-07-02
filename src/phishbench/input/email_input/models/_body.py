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
    if part.get_charset() is not None:
        return part.get_charset()
    content_type_string = part['Content-Type']
    if 'charset=' in content_type_string:
        return content_type_string.split('charset=')[1]


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
                basename, ext = part.get_filename().rsplit('.', maxsplit=1)
                self.file_extension_list.append(ext)

            if content_type == 'text/plain':
                self.__parse_text_part(part)
            elif content_type == 'text/html':
                self.is_html = True
                self.__parse_html_part(part)

    def __parse_text_part(self, part):
        try:
            payload = part.get_payload(decode=True)
            charset = get_charset(part)
            print(charset)
            if charset is not None:
                payload = payload.decode(charset).strip()
            else:
                payload = payload.decode().strip()
            if self.text:
                self.text += '\n'
                self.text += payload
            else:
                self.text = payload
        except UnicodeError:
            print("Failed To Decode")
            # We failed to decode the part
            self.defects.append(part)
            raw_data = part.get_payload()
            encoding = chardet.detect(raw_data)['encoding']
            if not encoding:
                # Fail to detect encoding
                encoding = 'utf-8'
            self.text = raw_data.decode(encoding=encoding, errors='replace')

    def __parse_html_part(self, part):
        charset = get_charset(part)
        print(charset)
        if charset is None:
            html = part.get_payload(decode=True).decode()
        else:
            html = part.get_payload(decode=True).decode(charset)
        cleaner = Cleaner()
        cleaner.javascript = True
        cleaner.style = True
        self.html = lxml.html.tostring(cleaner.clean_html(lxml.html.parse(StringIO(html))))
        if not self.text:
            soup = BeautifulSoup(self.html, 'html.parser')
            self.text = soup.get_text()
