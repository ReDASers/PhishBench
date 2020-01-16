import email
import os
from email.message import Message
from typing import List, Union

from .email_input import EmailHeader, EmailBody


def enumerate_folder_files(folder_path) -> List[str]:
    """
    Gets a list of text files in a folder. This method assumes a flat directory structure and dues not
    look in any sub-folders.
    Parameters
    ----------
    folder_path : str
        The path to the folder to enumerate.
    Returns
    -------
        A list containing the paths to every text folder in the directory.
    """
    # assumes a flat directory structure
    files = filter(lambda x: x.endswith('.txt'), os.listdir(folder_path))
    paths = map(lambda x: os.path.join(folder_path, x), files)
    return list(paths)


def read_email_from_file(file_path: str) -> Message:
    """
    Reads a email from a text file
    Parameters
    ----------
    file_path: str
        The path of the email to read
    Returns
    -------
    email.message.Message:
        A Message object representing the email.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return email.message_from_file(f)


def read_dataset_email(folder_path: str) -> Union[List[EmailBody], List[EmailHeader], List[Message], List[str]]:
    """

    Parameters
    ----------
    folder_path

    Returns
    -------

    """
    files = enumerate_folder_files(folder_path)
    emails = [read_email_from_file(f) for f in files]
    headers = [EmailHeader(msg) for msg in emails]
    bodies = [EmailBody(msg) for msg in emails]
    return bodies, headers, emails, files
