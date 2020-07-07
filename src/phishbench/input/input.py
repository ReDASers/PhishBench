import email
import glob
import os
import os.path
from email.message import Message
from typing import List, Union, Dict

import chardet

from .email_input.models import EmailMessage
from .url_input import URLData
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
    Reads a email from a file
    Parameters
    ----------
    file_path: str
        The path of the email to read
    Returns
    -------
    email.message.Message:
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
            # print('Falling back to string for {}'.format(file_name))
            text = f.read()
            msg = email.message_from_string(text)
    except UnicodeError:
        pass
    return msg


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


def read_dataset_email(folder_path: str) -> Union[List[EmailMessage], List[Message], List[str]]:
    """

    Parameters
    ----------
    folder_path

    Returns
    -------

    """
    files = enumerate_folder_files(folder_path)
    emails = [read_email_from_file(f) for f in files]
    emails_parsed = [EmailMessage(msg) for msg in emails]
    return emails_parsed, emails, files


def read_urls_from_file(file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("{} not found!".format(file_path))
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines


def read_dataset_url(dataset_path: str, download_url: bool, remove_dup: bool = True) -> List[URLData]:
    if os.path.isdir(dataset_path):
        corpus_files = enumerate_folder_files(dataset_path)
    else:
        corpus_files = [dataset_path]
    raw_urls = []
    for file_path in corpus_files:
        raw_urls.extend(read_urls_from_file(file_path))
    if remove_dup:
        old_len = len(raw_urls)
        raw_urls = list(set(raw_urls))
        Globals.logger.info("Removed %d duplicates", old_len - len(raw_urls))
    urls = []
    bad_url_list = []
    for raw_url in raw_urls:
        try:
            url_obj = URLData(raw_url, download_url)
            urls.append(url_obj)
        except Exception as e:
            Globals.logger.warning(
                "Exception while loading url %s", raw_url)
            Globals.logger.exception(e)
            bad_url_list.append(raw_url)
    return urls, bad_url_list
