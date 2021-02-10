from ._body import EmailBody
from ._header import EmailHeader


class EmailMessage:
    """
    A parsed email.

    Attributes
    ----------

    raw_message : email.message.Message
        The raw email message object
    header: EmailHeader
        The header of the email.
    body: EmailBody
        The body of the email
    """
    def __init__(self, msg):
        """
        Constructs an EmailMessage with a raw email.

        Parameters
        ==========
        msg: email.message.Message
            The raw email to parse
        """
        self.raw_message = msg
        self.body: EmailBody = EmailBody(msg)
        self.header: EmailHeader = EmailHeader(msg)
