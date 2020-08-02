import numpy as np
import tensorflow.keras as keras
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array

from ..core import BaseClassifier


def _build_model(n_features):
    model = keras.models.Sequential(
        layers=[
            keras.layers.Dense(80, activation='relu', input_dim=n_features),
            keras.layers.Dropout(rate=.3),
            keras.layers.Dense(1, activation='sigmoid')
        ]
    )
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class FeedForwardNN(BaseClassifier):

    def __init__(self, io_dir):
        super().__init__(io_dir, "FeedForwardNN.h5")

    def fit(self, x, y):
        n_features = x.shape[1]
        self.clf = _build_model(n_features)
        early_stopping = keras.callbacks.EarlyStopping(patience=3)
        self.clf.fit(x, y, epochs=150, batch_size=100, verbose=0, callbacks=[early_stopping])

    def predict_proba(self, x):
        if not self.clf:
            msg = "This {}} instance is not fitted yet. Call 'fit' with " \
                  "appropriate arguments before using this estimator.".format(type(self).__name__)

            raise NotFittedError(msg)
        x = check_array(x, accept_sparse=True)
        prob_pos = self.clf.predict(x)
        return prob_pos.flatten()

    def predict(self, x):
        return np.round(self.predict_proba(x))

    def load_model(self):
        self.clf = keras.models.load_model(self.model_path)

    def save_model(self):
        assert self.clf is not None, "Classifier must be trained first"
        self.clf.save(self.model_path)
