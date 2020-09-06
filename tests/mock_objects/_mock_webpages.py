import os
import pathlib

from bs4 import BeautifulSoup


def get_webpage(filename: str) -> str:
    """
    Gets the contents of mock webpages as a string.
    Parameters
    ----------
    filename:
        The name of the file relative to `resources/mock_webpages`
    Returns
    -------
    The contents of the file as a str.
    """
    current_file_folder = pathlib.Path(__file__).parent.absolute()
    test_file = os.path.join(current_file_folder, 'resources/mock_webpages', filename)
    with open(test_file) as f:
        return f.read()


def get_soup(filename: str) -> BeautifulSoup:
    """
    Gets a `BeautifulSoup` object from the
    Parameters
    ----------
    filename
        The name of the file relative to `resources/mock_webpages`
    Returns
    -------
        A `BeautifulSoup` object with the html5lib parser
    """
    return BeautifulSoup(get_webpage(filename), features="html5lib")
