import os
import pathlib

from bs4 import BeautifulSoup


def get_soup(filename) -> BeautifulSoup:
    current_file_folder = pathlib.Path(__file__).parent.absolute()
    test_file = os.path.join(current_file_folder, 'resources/mock_webpages', filename)
    with open(test_file) as f:
        soup = BeautifulSoup(f.read(), features="lxml")
    return soup

# def get_mock_urldata(filename) -> URLData:
#     current_file_folder = pathlib.Path(__file__).parent.absolute()
