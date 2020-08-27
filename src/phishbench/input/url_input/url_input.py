"""
This module handles file IO for URL datasets
"""
import os
from io import TextIOBase
from typing import Tuple
from typing import Union, List

from ._url_data import URLData
from ..input_utils import enumerate_folder_files, remove_duplicates
from ...utils import phishbench_globals


def read_dataset_url(dataset_path: str, download_url: bool, remove_dup: bool = True) -> Tuple[List[URLData], List[str]]:
    """
    Reads in a dataset of URLs from dataset_path
    :param dataset_path: The location of the dataset to read from. This can either be a folder or a file.
    :param download_url: Whether or not to download the urls
    :param remove_dup: Whether or not to remove duplicates.
    :return: A list of URLData objects representing the dataset
    """
    if not dataset_path:
        raise ValueError("dataset_path must be provided!")

    if os.path.isdir(dataset_path):
        corpus_files = enumerate_folder_files(dataset_path)
    elif os.path.exists(dataset_path):
        # dataset_path is a file
        corpus_files = [dataset_path]
    else:
        raise ValueError("{} does not exist.".format(dataset_path))

    raw_urls = []
    for file_path in corpus_files:
        raw_urls.extend(read_urls_from_file(file_path))

    if remove_dup:
        raw_urls = remove_duplicates(raw_urls)

    urls = []
    bad_url_list = []
    for raw_url in raw_urls:
        try:
            url_obj = URLData(raw_url, download_url)
            urls.append(url_obj)
        # pylint: disable=broad-except
        except Exception as e:
            phishbench_globals.logger.warning(
                "Exception while loading url %s", raw_url)
            phishbench_globals.logger.exception(e)
            bad_url_list.append(raw_url)

    return urls, bad_url_list


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
