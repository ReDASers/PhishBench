import os
import re
from email.message import Message

import lxml
from bs4 import BeautifulSoup
from lxml.html.clean import Cleaner

from ...utils import Globals

HEX_REGEX = re.compile(r"0x[0-9a-f]*?,?", flags=re.IGNORECASE | re.MULTILINE)
UNDERSCORE_REGEX = re.compile(r"_+", flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)


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
        body_text, body_html, soup, is_html, num_attachment, content_disposition_list, content_type_list, \
            content_transfer_encoding_list, file_extension_list, charset_list = EmailBody.extract_body(msg)
        self.text = body_text
        self.html = body_html
        self.is_html = is_html
        self.num_attachment = num_attachment
        self.content_disposition_list = content_disposition_list
        self.content_type_list = content_type_list
        self.content_transfer_encoding_list = content_transfer_encoding_list
        self.file_extension_list = file_extension_list
        self.charset_list = charset_list

    # Copied from original PhishBench code
    @classmethod
    def extract_body(cls, msg: Message):
        """
        Extracts the body from a Message
        Parameters
        ----------
        msg : email.message.Message
            the raw email to extract from
        Returns
        -------

        """
        is_html = False
        body_text = ''
        body_html = ''
        soup = None
        content_type_list = []
        content_disposition_list = []
        num_attachment = 0
        charset_list = []
        content_transfer_encoding_list = []
        file_extension_list = []
        for part in msg.walk():
            content_type = part.get_content_type()
            content_type_list.append(content_type)
            content_disposition = str(part.get_content_disposition())
            content_disposition_full = str(part.get('Content-Disposition'))
            filename = re.findall(r'(?!filename=)".*"', content_disposition_full)
            if filename:
                file_extension = os.path.splitext(filename[0])[1]
                file_extension_list.append(file_extension)
            content_disposition_list.append(content_disposition)
            if 'attachment' in content_disposition:
                num_attachment = +1
            if part.get_content_charset():
                charset_list.append(part.get_content_charset())
            c_transfer = str(part.get('Content-Transfer-Encoding'))
            content_transfer_encoding_list.append(c_transfer)
            if content_type == 'text/plain':
                Globals.logger.debug("text/plain loop")
                try:
                    body_text = part.get_payload(decode=True).decode(part.get_content_charset())
                    body_text = UNDERSCORE_REGEX.sub('', body_text)
                    body_text = body_text.strip()
                except Exception as e:
                    Globals.logger.debug('Exception: {}'.format(e))
                    Globals.logger.debug('Exception Handled')
                    body_text = part.get_payload(decode=False)
                    body_text = UNDERSCORE_REGEX.sub('', body_text)  # decode
            elif content_type == 'text/html':
                Globals.logger.debug("text/html loop")
                try:
                    html = part.get_payload(decode=True).decode(part.get_content_charset())
                except Exception as e:
                    Globals.logger.debug('Exception: {}'.format(e))
                    Globals.logger.debug('Exception Handled')
                    html = part.get_payload(decode=False)
                # Strips style and Javascript from html
                cleaner = Cleaner()
                cleaner.javascript = True
                cleaner.style = True
                html = lxml.html.tostring(cleaner.clean_html(lxml.html.parse(html)))
                soup = BeautifulSoup(html, 'html.parser')
                if body_text == '':
                    body_text = soup.text
                    body_text = HEX_REGEX.sub('', body_text)
                    body_text = UNDERSCORE_REGEX.sub('', body_text)
                body_html = str(soup)
                is_html = True
            else:
                Globals.logger.debug("else loop")
                body_text = part.get_payload(decode=False)

        return body_text, body_html, soup, is_html, num_attachment, content_disposition_list, \
            content_type_list, content_transfer_encoding_list, file_extension_list, charset_list
