from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

from ..core import BaseClassifier


class KNN(BaseClassifier):

    def __init__(self, io_dir):
        super().__init__(io_dir, "model_knn.pkl")

    def fit(self, x, y):
        self.clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                        metric='minkowski', metric_params=None, n_jobs=-1)
        self.clf.fit(x, y)

    def param_search(self, x, y):
        param_distributions = {'n_neighbors': range(3, 11, 2),
                               'leaf_size': range(20, 40),
                               'p': range(1, 5)
                               }
        clf = KNeighborsClassifier()
        cv_clf = RandomizedSearchCV(clf, param_distributions, n_iter=100, n_jobs=-1, pre_dispatch='2*n_jobs')
        self.clf = cv_clf.fit(x, y).best_estimator_
        return self.clf.get_params()
