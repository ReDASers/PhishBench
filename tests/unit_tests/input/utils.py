import email
from os import path


def get_relative_path(filename):
    current_loc = path.dirname(path.abspath(__file__))
    loc = path.join(current_loc, filename)
    return loc


def get_email(filename):
    loc = get_relative_path(filename)
    with open(loc, 'r') as file:
        return email.message_from_file(file)
