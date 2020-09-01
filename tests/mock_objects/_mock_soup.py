import os
import pathlib

from bs4 import BeautifulSoup


def get_html(filename: str) -> str:
    current_file_folder = pathlib.Path(__file__).parent.absolute()
    test_file = os.path.join(current_file_folder, 'resources/mock_webpages', filename)
    with open(test_file) as f:
        return f.read()


def get_soup(filename: str) -> BeautifulSoup:
    return BeautifulSoup(get_html(filename), features="lxml")
