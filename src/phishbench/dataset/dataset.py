import os
import sys

import joblib

from ..utils import Globals

sys.path.append('../')


def load_dataset(load_train=True, load_test=False):
    y_test = None
    X_test = None
    X_train = None
    y_train = None
    vectorizer_train = None
    vectorizer_test = None
    if Globals.config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
        email_train_dir = os.path.join(Globals.args.output_input_dir, "Emails_Training")
        vectorizer_train = joblib.load(os.path.join(email_train_dir, "vectorizer.pkl"))
        if load_train:
            X_train = joblib.load(os.path.join(email_train_dir, "X_train.pkl"))
            y_train = joblib.load(os.path.join(email_train_dir, "y_train.pkl"))

        try:
            if load_test:
                if Globals.config["Classification"]["Attack Features"] == "True":
                    with open(os.path.join(email_train_dir, "features.txt"), 'r') as f:
                        dict_test = eval(f.read())
                    X_test = vectorizer_train.fit_transform(dict_test)
                    y_test = joblib.load(os.path.join(email_train_dir, "y_test.pkl"))
                else:
                    email_test_dir = os.path.join(Globals.args.output_input_dir, "Emails_Testing")
                    vectorizer_test = joblib.load(os.path.join(email_test_dir, "vectorizer.pkl"))
                    X_test = joblib.load(os.path.join(email_test_dir, "X_test.pkl"))
                    y_test = joblib.load(os.path.join(email_test_dir, "y_test.pkl"))
        except FileNotFoundError as ex:
            Globals.logger.warn("Test files not found {}".format(ex))

    elif Globals.config["Email or URL feature Extraction"]["extract_features_URLs"] == "True":
        url_train_dir = os.path.join(Globals.args.output_input_dir, "URLs_Training")
        url_test_dir = os.path.join(Globals.args.output_input_dir, "URLs_Testing")
        vectorizer_train = joblib.load(os.path.join(url_train_dir, "vectorizer.pkl"))
        if load_train:
            X_train = joblib.load(os.path.join(url_train_dir, "X_train.pkl"))
            y_train = joblib.load(os.path.join(url_train_dir, "y_train.pkl"))
        try:
            if load_test:
                if Globals.config["Classification"]["Attack Features"] == "True":
                    with open(os.path.join(url_train_dir, "features.txt"), 'r') as f:
                        dict_test = eval(f.read())
                    X_test = vectorizer_test.fit_transform(dict_test)
                    y_test = joblib.load(os.path.join(url_train_dir, "y_test.pkl"))
                else:
                    X_test = joblib.load(os.path.join(url_test_dir, "X_test.pkl"))
                    y_test = joblib.load(os.path.join(url_test_dir, "y_test.pkl"))
                    vectorizer_test = joblib.load(os.path.join(url_test_dir, "vectorizer.pkl"))
        except FileNotFoundError as ex:
            Globals.logger.warn("Test files not found {}".format(ex))

    return X_train, y_train, X_test, y_test, vectorizer_train, vectorizer_test
