"""
This module handles file IO for URL datasets
"""
from typing import Union, List
from io import TextIOBase


def read_urls_from_file(f: Union[TextIOBase, str]) -> List[str]:
    """
    Reads urls from a text file.
    :param f: A file-like object to read from or a path to a text file.
    :return: A list of urls
    """
    close = False
    if isinstance(f, str):
        f = open(f, 'r')
        close = True

    lines = f.readlines()
    lines = [x.strip() for x in lines]

    if close:
        f.close()

    return lines
