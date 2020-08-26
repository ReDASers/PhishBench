import glob
import os
import os.path

from typing import List, Tuple

from .url_input import URLData
from .url_input.url_io import read_urls_from_file
from ..utils import phishbench_globals


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


def remove_duplicates(values: List):
    """
    Removes duplicates from a list
    :param values: The list to remove duplicates from
    :return: A new list without duplicates.
    """
    old_len = len(values)
    clean = list(set(values))
    phishbench_globals.logger.info("Removed %d duplicates", old_len - len(clean))
    return clean


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
        except Exception as e:
            phishbench_globals.logger.warning(
                "Exception while loading url %s", raw_url)
            phishbench_globals.logger.exception(e)
            bad_url_list.append(raw_url)

    return urls, bad_url_list
