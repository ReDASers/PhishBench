import os

import joblib


class BaseClassifier:

    def __init__(self, io_dir, save_file):
        self.io_dir = io_dir
        self.model_path: str = os.path.join(self.io_dir, save_file)
        self.clf = None

    def fit(self, x, y):
        """
        Trains the classifier. If being used as a wrapper for a scikit-learn style classifier, then implementations of
        this function should store the trained underlying classifier in `self.clf`. Other implementations should
        also override predict and predict_proba
        Parameters
        ----------
        x: array-like or sparse matrix of shape (n,f)
            Training vectors, where n is the number of samples and f is the number of features.
        y: array-like of shape (n)
            Target values, with 0 being legitimate and 1 being phishing
        Returns
        -------
            None
        """

    def fit_weighted(self, x, y):
        print("{} does not support weighted training. Performing regular training.".format(self.name))
        self.fit(x, y)

    def param_search(self, x, y):
        """
        Performs parameter search to find the best parameters.
        Parameters
        ----------
        x: array-like or sparse matrix of shape (n,f)
            Training vectors, where n is the number of samples and f is the number of features.
        y: array-like of shape (n)
            Target values, with 0 being legitimate and 1 being phishing
        Returns
        -------
        dict:
            The best parameters.
        """
        print("{} does not support parameter search. Performing regular training.".format(self.name))
        self.fit(x, y)

    def predict(self, x):
        """
        Parameters
        ----------
        x: array-like or sparse matrix of shape (n,f)
            Test vectors, where n is the number of samples and f is the number of features
        Returns
        -------
        array-like of shape (n)
            The predicted class values
        """
        assert self.clf is not None, "Classifier must be trained first"
        return self.clf.predict(x)

    def predict_proba(self, x):
        """
        Parameters
        ----------
        x: array-like or sparse matrix of shape (n,f)
            Test vectors, where n is the number of samples and f is the number of features
        Returns
        -------
        array-like of shape (n)
            The probability of each test vector being phish
        """
        assert self.clf is not None, "Classifier must be trained first"
        return self.clf.predict_proba(x)[:, 1]

    def load_model(self):
        """
        Loads the model from `self.io_dir`
        Returns
        -------
            None
        """
        self.clf = joblib.load(self.model_path)

    def save_model(self):
        """
        Saves the model to `self.io_dir`
        Returns
        -------
            None
        """
        assert self.clf is not None, "Classifier must be trained first"
        joblib.dump(self.clf, self.model_path)

    @property
    def name(self):
        return type(self).__name__
