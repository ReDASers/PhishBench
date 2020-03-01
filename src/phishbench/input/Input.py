import email
import os
from email.message import Message
from typing import List, Union
import glob

from .email_input import EmailHeader, EmailBody


def enumerate_folder_files(folder_path) -> List[str]:
    """
    Recursively searches a folder for .txt files
    Parameters
    ----------
    folder_path : str
        The path to the folder to search.
    Returns
    -------
        A list containing the paths to every text enumerate in the directory.
    """
    glob_path = os.path.join(folder_path, "**/*.txt")
    return glob.glob(glob_path,recursive=True)


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
