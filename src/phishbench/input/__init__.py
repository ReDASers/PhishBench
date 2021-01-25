"""
This module loads datasets from raw emails and URLs. It is split into two sub-modules. The `email_input` submodule
handles email datasets, and the `url_input` submodule handles URL datasets.

In addition, this module contains `read_train_set` and `read_test_set` functions which uses the relevant submodule to
read in datasets according to the global configuration.
"""
from . import settings
from ._input import read_test_set, read_train_set
from .email_input import EmailBody, EmailHeader, EmailMessage
from .url_input import URLData
