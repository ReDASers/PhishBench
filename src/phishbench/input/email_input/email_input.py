"""
This module contains the implementations for email input functions
"""
import email
import traceback
from email.message import Message
from typing import List, Tuple

import chardet
from tqdm import tqdm

from .models import EmailMessage
from ..input import enumerate_folder_files


def read_dataset_email(folder_path: str) -> Tuple[List[EmailMessage], List[str]]:
    """

    Parameters
    ----------
    folder_path : str
        The path to the folder you want to read

    Returns
    -------
    emails: List[EmailMessage]
        The parsed emails

    files: List[str]
        The paths of the files loaded
    """
    files = enumerate_folder_files(folder_path)
    loaded_files = []
    emails_parsed = []
    for f in tqdm(files):
        try:
            msg = EmailMessage(read_email_from_file(f))
            emails_parsed.append(msg)
            loaded_files.append(f)
        except Exception:
            print("\n", f)
            traceback.print_exc()

    return emails_parsed, loaded_files


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
