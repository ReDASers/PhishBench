from ._body import EmailBody
from ._header import EmailHeader


class EmailMessage:

    def __init__(self, msg):
        self.raw_message = msg
        self.body: EmailBody = EmailBody(msg)
        self.header: EmailHeader = EmailHeader(msg)
