import email
from email.message import Message

import chardet


def read_email_from_file(file_path: str) -> Message:
    """
    Reads a email from a file
    Parameters
    ----------
    file_path: str
        The path of the email to read

    Returns
    -------
    msg: email.message.Message
        A Message object representing the email.
    """
    with open(file_path, 'rb') as f:
        binary_data = f.read()
        msg = email.message_from_bytes(binary_data)
    try:
        for part in msg.walk():
            if not part.is_multipart():
                str(part)
    except (LookupError, KeyError):
        # LookupError -> No charset
        # KeyError -> No content transfer encoding
        encoding = chardet.detect(binary_data)['encoding']
        if not encoding:
            encoding = None
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            # print('Falling back to string for {}'.format(file_path))
            text = f.read()
            msg = email.message_from_string(text)
    except UnicodeError:
        pass
    return msg
