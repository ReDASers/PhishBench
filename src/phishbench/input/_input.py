"""
Contains implementations of input library functions
"""
from typing import Union, List, Tuple

from . import email_input
from . import settings
from . import url_input
from .. import settings as pb_settings


def read_train_set(download_url: bool = False) -> Tuple[
        Union[List[url_input.URLData], List[email_input.EmailMessage]],
        List[int]
        ]:
    """
    Reads in the training set according to the configuration file.

    Parameters
    ----------
    download_url: bool
        When loading a URL dataset, whether or not to download the websites pointed to by the URLs. Ignored when
        loading a Email dataset.

    Returns
    -------
    data: A list of `URLData` or `EmailMessage` objects
        The read-in data.
    labels: List[int]
        A list of labels. `0` is legitimate and `1` is phish
    """
    if pb_settings.mode() == 'URL':
        legit, _ = url_input.read_dataset_url(settings.train_legit_path(), download_url)
        phish, _ = url_input.read_dataset_url(settings.train_phish_path(), download_url)
    else:
        legit, _ = email_input.read_dataset_email(settings.train_legit_path())
        phish, _ = email_input.read_dataset_email(settings.train_phish_path())
    data = legit + phish
    labels = [0] * len(legit) + [1] * len(phish)
    return data, labels


def read_test_set(download_url: bool = False) -> Tuple[
        Union[List[url_input.URLData], List[email_input.EmailMessage]],
        List[int]
        ]:
    """
    Reads in the test set according to the configuration file.

    Parameters
    ----------
    download_url: bool
        When loading a URL dataset, whether or not to download the websites pointed to by the URLs. Ignored when
        loading a Email dataset.

    Returns
    -------
    data: URl mode: - A list of `URLData` or `EmailMessage` objects
        The read-in data.
    labels: List[int]
        A list of labels. `0` is legitimate and `1` is phish
    """
    if pb_settings.mode() == 'URL':
        legit, _ = url_input.read_dataset_url(settings.test_legit_path(), download_url)
        phish, _ = url_input.read_dataset_url(settings.test_phish_path(), download_url)
    else:
        legit = email_input.read_dataset_email(settings.test_legit_path())
        phish = email_input.read_dataset_email(settings.test_phish_path())
    data = legit + phish
    labels = [0] * len(legit) + [1] * len(phish)
    return data, labels
