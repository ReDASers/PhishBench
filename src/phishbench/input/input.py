import email
import os
from email.message import Message
from typing import List, Union, Dict
import glob

from .email_input import EmailHeader, EmailBody
from ..utils import Globals


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
    return glob.glob(glob_path, recursive=True)


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


def read_corpus(corpus_files, encoding='utf-8') -> Dict[str, str]:
    """
    Reads a corpus
    Parameters
    ----------
    corpus_files
    encoding

    Returns
    -------
        A dictionary with the file paths as the key and the file contents as the values
    """
    corpus = {}
    for filepath in corpus_files:
        try:
            with open(filepath, 'r', encoding=encoding, errors='ignore') as file:
                corpus[filepath] = file.read()
        except Exception as e:
            Globals.logger.warning("exception: %s", e)
    return corpus


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