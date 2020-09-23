"""
Handles email input
"""
from .email_input import read_dataset_email
from .email_input import read_email_from_file
from .models import EmailBody, EmailMessage, EmailHeader
from .models.date_parse import parse_email_datetime
