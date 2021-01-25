"""
This module loads datasets from raw emails and URLs. 


"""
from . import settings
from ._input import read_test_set, read_train_set
from .email_input import EmailBody, EmailHeader, EmailMessage
from .url_input import URLData
