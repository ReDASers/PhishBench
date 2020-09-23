"""
Handles user interaction with PhishBench
"""
import sys

from phishbench.classification import settings as classification_setings
from phishbench.utils.phishbench_globals import config
from phishbench.input import settings as input_settings


def query_yes_no(question, default="yes"):
    """
    Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        if choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def confirmation(ignore_confirmation=False):
    """
    Displays settings and propts user for confirmation.
    Parameters
    ----------
    ignore_confirmation:
        Whether to automatically return `True`

    Returns
    -------
    `True` if `ignore_confirmation` is `True` or the user confirms that the settings are correct. `False` otherwise.
    """
    print("##### Review of Options:")
    if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
        print("Running Email Mode")
    elif config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
        print("Running URL Mode")

    print("###Paths to datasets:")
    print("Legitimate Dataset (Training): {}".format(input_settings.train_legit_path()))
    print("Phishing Dataset (Training):: {}".format(input_settings.train_phish_path()))
    print("Legitimate Dataset (Testing): {}".format(input_settings.test_legit_path()))
    print("Phishing Dataset (Testing): {}".format(input_settings.test_phish_path()))

    if config["Extraction"]["feature extraction"] == "True":
        print("\nRun the Feature Extraction: {}".format(config["Extraction"]["feature extraction"]))
        print("\nFeature Extraction for Training Data: {}".format(config["Extraction"]["training dataset"]))
        print("\nFeature Extraction for Testing Data: {}".format(config["Extraction"]["testing dataset"]))
    else:
        print("\nRun the Feature Extraction: {}".format(config["Extraction"]["feature extraction"]))

    print("\nFeature Selection: {}".format(config['Feature Selection']['select best features']))
    print("\nRun the classifiers: {}".format(classification_setings.run_classifiers()))
    print("\n")
    if ignore_confirmation:
        return True
    return query_yes_no("Do you wish to continue?")
