"""
Contains the Email Body TF-IDF extractor
"""
from typing import List
import pickle

import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ...reflection import FeatureMC, FeatureType
from ....input import EmailMessage, EmailBody


class EmailBodyTfidf(metaclass=FeatureMC):
    """
    TF-IDF world-level vectors of the email bodies
    """
    config_name = 'email_body_tfidf'
    feature_type = FeatureType.EMAIL_BODY
    default_value = None

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),
                                                min_df=0, stop_words='english', sublinear_tf=True)

    def fit(self, corpus: List[EmailMessage], labels):
        """
        Fits the TF-IDF feature

        Parameters
        ----------
        corpus: List[EmailMessage]
            The training corpus
        labels:
            Ignored
        """
        # pylint: disable=unused-argument
        bodies = [msg.body.text for msg in corpus]
        self.tfidf_vectorizer.fit(bodies)
        vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        self.default_value = scipy.sparse.csr_matrix((1, vocab_size))

    def extract(self, body: EmailBody):
        """
        Extracts a TF-IDF vector

        Parameters
        ----------
        body: EmailBody
            The `EmailBody` to extract the vector from
        """
        x = [body.text]
        return self.tfidf_vectorizer.transform(x)

    def load_state(self, filename):
        """
        Loads the TF-IDF vectorizer from a file

        Parameters
        ----------
        filename: str
            The name of the file to load the state from
        """
        with open(filename, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        self.default_value = scipy.sparse.csr_matrix((1, vocab_size))

    def save_state(self, filename):
        """
        Saves the TF-IDF vectorizer to a file
        Parameters
        ----------
        filename: str
            The name of the file to save the state to
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
