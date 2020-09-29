"""
Handles user interaction with PhishBench
"""
import sys

import phishbench.settings
from phishbench.classification import settings as classification_setings
from phishbench.input import settings as input_settings
from phishbench.utils.phishbench_globals import config


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
        sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


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

    print("\nPhishBench is running in {} mode.\n".format(phishbench.settings.mode()))

    if phishbench.settings.feature_extraction():
        print("Performing feature extraction with")
        if config["Extraction"].getboolean("training dataset"):
            print("\tLegitimate Dataset (Training): {}".format(input_settings.train_legit_path()))
            print("\tPhishing Dataset (Training):: {}".format(input_settings.train_phish_path()))
        if config["Extraction"].getboolean("testing dataset"):
            print("\tLegitimate Dataset (Testing): {}".format(input_settings.test_legit_path()))
            print("\tPhishing Dataset (Testing): {}".format(input_settings.test_phish_path()))
    else:
        print("Loading features from disk")

    if phishbench.settings.feature_selection():
        print("\nPerforming Feature selection")

    if phishbench.settings.classification():
        print("\nRunning classifiers")
        classification_section = config[classification_setings.CLASSIFIERS_SECTION]
        for classifier in classification_section.keys():
            if classification_section.getboolean(classifier):
                print(f'\t{classifier}')
    print("\n")
    if ignore_confirmation:
        return True
    return query_yes_no("Do you wish to continue?")
