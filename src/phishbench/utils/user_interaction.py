import sys

from . import Globals


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
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
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def Confirmation(ignore_confirmation=False):
    print("##### Review of Options:")
    if Globals.config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
        print("extract_features_emails = {}".format(Globals.config["Email or URL feature Extraction"]["extract_features_emails"]))
    elif Globals.config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
        print("extract_features_urls = {}".format(Globals.config["Email or URL feature Extraction"]["extract_features_urls"]))

    print("###Paths to datasets:")
    print("Legitimate Dataset (Training): {}".format(Globals.config["Dataset Path"]["path_legitimate_training"]))
    print("Phishing Dataset (Training):: {}".format(Globals.config["Dataset Path"]["path_phishing_training"]))
    print("Legitimate Dataset (Testing): {}".format(Globals.config["Dataset Path"]["path_legitimate_testing"]))
    print("Phishing Dataset (Testing): {}".format(Globals.config["Dataset Path"]["path_phishing_testing"]))

    print("\nRun Feature Ranking Only: {}".format(Globals.config["Feature Selection"]["Feature Ranking Only"]))
    if Globals.config["Extraction"]["feature extraction"]=="True":
        print("\nRun the Feature Extraction: {}".format(Globals.config["Extraction"]["feature extraction"]))
        print("\nFeature Extraction for Training Data: {}".format(Globals.config["Extraction"]["training dataset"]))
        print("\nFeature Extraction for Testing Data: {}".format(Globals.config["Extraction"]["testing dataset"]))
    else:
        print("\nRun the Feature Extraction: {}".format(Globals.config["Extraction"]["feature extraction"]))
    print("\nRun the classifiers: {}".format(Globals.config["Classification"]["Running the classifiers"]))
    print("\n")
    if ignore_confirmation:
        answer = True
    else:
        answer = query_yes_no("Do you wish to continue?")
    return answer
