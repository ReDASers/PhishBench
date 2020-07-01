from ._body import EmailBody
from ._header import EmailHeader


class EmailMessage:

    def __init__(self, msg):
        self.body = EmailBody(msg)
        self.header = EmailHeader(msg)
