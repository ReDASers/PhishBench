"""
This module contains the implementations for email input functions
"""
import email
import traceback
from email.message import Message
from email.header import Header
from typing import List, Tuple

import chardet
from tqdm import tqdm

from .models import EmailMessage
from ..input_utils import enumerate_folder_files


def read_dataset_email(folder_path: str) -> Tuple[List[EmailMessage], List[str]]:
    """
    Reads a folder of emails

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
    if not folder_path:
        raise ValueError("folder_path must be provided!")

    print(f"Loading emails from {folder_path}")

    files = enumerate_folder_files(folder_path)
    loaded_files = []
    emails_parsed = []
    for filename in tqdm(files):
        try:
            msg = EmailMessage(read_email_from_file(filename))
            emails_parsed.append(msg)
            loaded_files.append(filename)
        # pylint: disable=broad-except
        except Exception:
            print(f"\nFailed to parse {filename}\n")
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
        for header in msg:
            if isinstance(msg[header], Header):
                raise KeyError(f"Failed to parse {header}")
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
