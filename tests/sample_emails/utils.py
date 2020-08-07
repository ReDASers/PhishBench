import email
from os import path
import chardet


def get_relative_path(filename):
    current_loc = path.dirname(path.abspath(__file__))
    loc = path.join(current_loc, "Resources", filename)
    return loc


def get_binary_email(filename):
    loc = get_relative_path(filename)
    with open(loc, 'rb') as file:
        return email.message_from_binary_file(file)


def get_email_text(filename):
    loc = get_relative_path(filename)
    with open(loc, 'rb') as f:
        binary_data = f.read()
    encoding = chardet.detect(binary_data)['encoding'].lower()
    with open(loc, 'r', encoding=encoding) as file:
        contents = file.read()
    return email.message_from_string(contents)
