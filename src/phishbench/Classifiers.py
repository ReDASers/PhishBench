import os
import os.path
import re
import time

import joblib
import numpy as np
import sklearn
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
from numpy.random import RandomState
from scipy.stats import uniform
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import MLPRandomLayer

from . import Evaluation_Metrics
from . import Features
from .dataset import Imbalanced_Dataset
from .utils import Globals


####### Dataset (features for each item) X and Classess y (phish or legitimate)

def load_dictionary():
    list_dict_train = joblib.load('list_dict_train.pkl')
    list_dict_test = joblib.load('list_dict_test.pkl')
    vec = DictVectorizer()
    Sparse_Matrix_Features_train = vec.fit_transform(list_dict_train)
    Sparse_Matrix_Features_test = vec.transform(list_dict_test)

    labels_train = joblib.load('labels_train.pkl')
    labels_test = joblib.load('labels_test.pkl')
    # preprocessing
    return Sparse_Matrix_Features_train, labels_train, Sparse_Matrix_Features_test, labels_test


def fit_classifier(clf, X, y, X_train_balanced=None, y_train_balanced=None):
    start_time = time.time()
    if X_train_balanced is not None and y_train_balanced is not None:
        clf.fit(X_train_balanced, y_train_balanced)
    else:
        clf.fit(X, y)
    Globals.logger.info("Training Time = " + str(time.time() - start_time) + "s")


def SVM(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
    Globals.logger.info("SVM >>>>>>>")
    if clf is None:
        if Globals.config["Classifiers"]["weighted"] == "True":
            clf = svm.SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
                          decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                          max_iter=-1, probability=False, random_state=None, shrinking=True,
                          tol=0.001, verbose=False)
        else:
            """
            clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                   decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                   max_iter=-1, probability=False, random_state=None, shrinking=True,
                   tol=0.001, verbose=False)
            """
            from sklearn.svm import LinearSVC
            clf = LinearSVC()
        # clf = LinearSVC(penalty="l1",loss="hinge", dual=True, C=100, multi_class="crammer_singer",class_weight=None)
        if Globals.config["Evaluation Metrics"]["cross_validate"] == "True":
            score = Evaluation_Metrics.Cross_validation(clf, X, y)
            Globals.logger.info(score)
            return score, None
        if Globals.config["Evaluation Metrics"]["parameter_search"] == "True":
            param_distributions = {"penalty": ['l1', 'l2'], "loss": ['squared_hinge', 'hinge'], "dual": [True, False],
                                   "C": [0.1, 1, 10, 100, 1000], "multi_class": ['crammer_singer', 'ovr'],
                                   "class_weight": [None, 'balanced']}
            clf = RandomizedSearchCV(clf, param_distributions, n_iter=100, scoring=None, fit_params=None, n_jobs=-1,
                                     iid='warn', refit=True, cv=10, verbose=2, pre_dispatch='2*n_jobs',
                                     random_state=None, error_score=0, return_train_score='warn')
            best_model = clf.fit(X, y)
            print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
            print('Best loss:', best_model.best_estimator_.get_params()['loss'])
            print('Best dual:', best_model.best_estimator_.get_params()['dual'])
            print('Best C:', best_model.best_estimator_.get_params()['C'])
            print('Best multi_class:', best_model.best_estimator_.get_params()['multi_class'])
            print('Best class_weight:', best_model.best_estimator_.get_params()['class_weight'])
            return None, None
        else:
            fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
            # clf.fit(X, y)
            y_predict = clf.predict(X_test)
            eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
            return eval_metrics_SVM, clf
    else:
        y_predict = clf.predict(X_test)
        eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
        return eval_metrics_SVM, clf


######## Random Forest
def RandomForest(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
    Globals.logger.info("RF >>>>>>>")
    if clf is None:
        if Globals.config["Classifiers"]["weighted"] == "True":
            clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,
                                         oob_score=False, n_jobs=1,
                                         random_state=None, verbose=0, warm_start=False, class_weight='balanced')
        else:
            """
            clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1,
                random_state=None, verbose=0, warm_start=False, class_weight=None)
            """
            clf = RandomForestClassifier(n_estimators=80, criterion='gini', max_depth=90, min_samples_split=10,
                                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False,
                                         oob_score=False, n_jobs=-1,
                                         random_state=None, verbose=0, warm_start=False, class_weight=None)
        if Globals.config["Evaluation Metrics"]["cross_validate"] == "True":
            score = Evaluation_Metrics.Cross_validation(clf, X, y)
            Globals.logger.info(score)
            return score, None
        if Globals.config["Evaluation Metrics"]["parameter_search"] == "True":
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            param_distributions = {"n_estimators": [int(x) for x in np.linspace(start=10, stop=100, num=10)],
                                   "max_depth": max_depth, "min_samples_split": [2, 5, 10],
                                   "min_samples_leaf": [1, 2, 4], "max_features": ['auto', 'sqrt'],
                                   "bootstrap": [True, False], "class_weight": [None, 'balanced', 'balanced_subsample']}
            clf = RandomizedSearchCV(clf, param_distributions, n_iter=10, scoring=None, fit_params=None, n_jobs=None,
                                     iid='warn', refit=True, cv=10, verbose=3, pre_dispatch='2*n_jobs',
                                     random_state=None, error_score='raise-deprecating', return_train_score='warn')
            best_model = clf.fit(X, y)
            print('Best n_estimators:', best_model.best_estimator_.get_params()['n_estimators'])
            print('Best max_depth:', best_model.best_estimator_.get_params()['max_depth'])
            print('min_samples_split:', best_model.best_estimator_.get_params()['min_samples_split'])
            print('min_samples_leaf:', best_model.best_estimator_.get_params()['min_samples_leaf'])
            print('max_features:', best_model.best_estimator_.get_params()['max_features'])
            print('bootstrap:', best_model.best_estimator_.get_params()['bootstrap'])
            print('class_weight:', best_model.best_estimator_.get_params()['class_weight'])
            return None, None
        else:
            fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
            y_predict = clf.predict(X_test)
            eval_metrics_RF = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
            return eval_metrics_RF, clf
    else:
        y_predict = clf.predict(X_test)
        eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
        return eval_metrics_SVM, clf


###### Decition Tree
def DecisionTree(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
    Globals.logger.info("DT >>>>>>>")
    if clf is None:
        if Globals.config["Classifiers"]["weighted"] == "True":
            clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
                                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                         random_state=None, max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, min_impurity_split=None, class_weight='balanced',
                                         presort=False)
        else:
            """
            clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
            """
            clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=40, min_samples_split=2,
                                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                         random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                         min_impurity_split=None, class_weight=None, presort=False)
        if Globals.config["Evaluation Metrics"]["cross_validate"] == "True":
            score = Evaluation_Metrics.Cross_validation(clf, X, y)
            Globals.logger.info(score)
            return score, None
        if Globals.config["Evaluation Metrics"]["parameter_search"] == "True":
            # https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True).tolist() + np.linspace(2, 6, 5, endpoint=True,
                                                                                                dtype=int).tolist()
            min_sample_leaf = np.linspace(0.1, 0.5, 5, endpoint=True).tolist() + np.linspace(1, 5, 5, endpoint=True,
                                                                                             dtype=int).tolist()
            param_distributions = {"max_depth": max_depth, "min_samples_split": min_samples_split,
                                   "min_samples_leaf": min_sample_leaf, "max_features": ['auto', 'sqrt', None]}
            clf = RandomizedSearchCV(clf, param_distributions, n_iter=150, scoring=None, fit_params=None, n_jobs=None,
                                     iid='warn', refit=True, cv=10, verbose=2, pre_dispatch='2*n_jobs',
                                     random_state=None, error_score='raise-deprecating', return_train_score='warn')
            best_model = clf.fit(X, y)
            print('Best max_depth:', best_model.best_estimator_.get_params()['max_depth'])
            print('Best min_samples_split:', best_model.best_estimator_.get_params()['min_samples_split'])
            print('Best min_samples_leaf:', best_model.best_estimator_.get_params()['min_samples_leaf'])
            print('Best max_features:', best_model.best_estimator_.get_params()['max_features'])
            return None, None
        else:
            fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
            y_predict = clf.predict(X_test)
            eval_metrics_DT = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
            return eval_metrics_DT, clf
    else:
        y_predict = clf.predict(X_test)
        eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
        return eval_metrics_SVM, clf


##### Gaussian Naive Bayes
def GaussianNaiveBayes(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
    Globals.logger.info("GNB >>>>>>>")
    if clf is None:
        if Globals.config["Classifiers"]["weighted"] == "True":
            Globals.logger.warn("GaussianNaiveBayes does not support weighted classification")
            return
        clf = GaussianNB(priors=None, var_smoothing=1e-06)
        # clf = GaussianNB(priors=None)
        if Globals.config["Evaluation Metrics"]["cross_validate"] == "True":
            score = Evaluation_Metrics.Cross_validation(clf, X, y)
            Globals.logger.info(score)
            return score, None
        if Globals.config["Evaluation Metrics"]["parameter_search"] == "True":
            param_distributions = {"var_smoothing": [1e-09, 1e-08, 1e-07, 1e-06]}
            clf = RandomizedSearchCV(clf, param_distributions, n_iter=40, scoring=None, fit_params=None, n_jobs=None,
                                     iid='warn', refit=True, cv=10, verbose=0, pre_dispatch='2*n_jobs',
                                     random_state=None, error_score='raise-deprecating', return_train_score='warn')
            # X=X.toarray()
            best_model = clf.fit(X, y)
            print('Best var_smoothing:', best_model.best_estimator_.get_params()['var_smoothing'])
            return None, None
        else:
            # X=X.toarray()
            fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
            # X_test=X_test.toarray()
            y_predict = clf.predict(X_test)
            eval_metrics_NB = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
            return eval_metrics_NB, clf
    else:
        y_predict = clf.predict(X_test)
        eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
        return eval_metrics_SVM, clf


##### Multinomial Naive Bayes
def MultinomialNaiveBayes(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
    Globals.logger.info("MNB >>>>>>>")
    if clf is None:
        if Globals.config["Classifiers"]["weighted"] == "True":
            Globals.logger.warn("MultinomialNaiveBayes does not support weighted classification")
            return
        # clf=MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
        clf = MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None)
        if Globals.config["Evaluation Metrics"]["cross_validate"] == "True":
            score = Evaluation_Metrics.Cross_validation(clf, X, y)
            Globals.logger.info(score)
            return score, None
        if Globals.config["Evaluation Metrics"]["parameter_search"] == "True":
            # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74?gi=971100db22f7
            param_distributions = {"alpha": [0.1, 0.5, 1]}
            clf = RandomizedSearchCV(clf, param_distributions, n_iter=10, scoring=None, fit_params=None, n_jobs=None,
                                     iid='warn', refit=True, cv=10, verbose=0, pre_dispatch='2*n_jobs',
                                     random_state=None, error_score='raise-deprecating', return_train_score='warn')
            best_model = clf.fit(X, y)
            print('Best Alpha:', best_model.best_estimator_.get_params()['alpha'])
            return None, None
        else:
            fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
            y_predict = clf.predict(X_test)
            eval_metrics_MNB = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
            return eval_metrics_MNB, clf
    else:
        y_predict = clf.predict(X_test)
        eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
        return eval_metrics_SVM, clf


##### Logistic Regression
def LogisticRegression(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
    Globals.logger.info("LR >>>>>>>")
    if clf is None:
        if Globals.config["Classifiers"]["weighted"] == "True":
            clf = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                                                          fit_intercept=True, intercept_scaling=1,
                                                          class_weight='balanced', random_state=None,
                                                          solver='liblinear', max_iter=100, multi_class='ovr',
                                                          verbose=0, warm_start=False, n_jobs=1)
        else:
            # clf=sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
            clf = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=4, fit_intercept=True,
                                                          intercept_scaling=1, class_weight=None, random_state=None,
                                                          solver='sag', max_iter=100, multi_class='ovr', verbose=0,
                                                          warm_start=False, n_jobs=1)
        if Globals.config["Evaluation Metrics"]["cross_validate"] == "True":
            score = Evaluation_Metrics.Cross_validation(clf, X, y)
            Globals.logger.info(score)
            return score, None
        if Globals.config["Evaluation Metrics"]["parameter_search"] == "True":
            penalty = ['l1', 'l2']
            C = [1, 2, 3, 4]
            solver = ['warn', 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            # hyperparameters = dict(penalty=penalty, solver=solver, C=C)
            hyperparameters = dict(solver=solver, C=C)
            clf = RandomizedSearchCV(clf, hyperparameters, random_state=1, n_iter=100, cv=10, verbose=5, n_jobs=1)
            best_model = clf.fit(X, y)
            print('Best solver:', best_model.best_estimator_.get_params()['solver'])
            print('Best C:', best_model.best_estimator_.get_params()['C'])
            return None, None
        if Globals.config["Evaluation Metrics"]["parameter_search"] == "True":
            penalty = ['l1', 'l2']
            C = uniform(loc=0, scale=4)
            solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            # hyperparameters = dict(penalty=penalty, solver=solver, C=C)
            hyperparameters = dict(penalty=penalty, C=C)
            clf = RandomizedSearchCV(clf, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=5, n_jobs=1)
            best_model = clf.fit(X, y)
            print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
            print('Best C:', best_model.best_estimator_.get_params()['C'])
            return None, None
        else:
            fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
            y_predict = clf.predict(X_test)
            eval_metrics_LR = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
            return eval_metrics_LR, clf
    else:
        y_predict = clf.predict(X_test)
        eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
        return eval_metrics_SVM, clf


##### ELM
def ELM(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
    Globals.logger.info("ELM >>>>>>>")
    if clf is None:
        if Globals.config["Classifiers"]["weighted"] == "True":
            Globals.logger.warn("kNearestNeighbor does not support weighted classification")
            return

        srhl_tanh = MLPRandomLayer(n_hidden=10, activation_func='tanh')
        clf = GenELMClassifier(hidden_layer=srhl_tanh)
        if Globals.config["Evaluation Metrics"]["cross_validate"] == "True":
            score = Evaluation_Metrics.Cross_validation(clf, X, y)
            Globals.logger.info(score)
            return score, None
        else:
            fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
            y_predict = clf.predict(X_test)
            eval_metrics_ELM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
            return eval_metrics_ELM, clf
    else:
        y_predict = clf.predict(X_test)
        eval_metrics_ELM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
        return eval_metrics_ELM, clf


##### k-Nearest Neighbor
def kNearestNeighbor(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
    Globals.logger.info("KNN >>>>>>>")
    if clf is None:
        if Globals.config["Classifiers"]["weighted"] == "True":
            Globals.logger.warn("kNearestNeighbor does not support weighted classification")
            return

        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=-1, )
        if Globals.config["Evaluation Metrics"]["cross_validate"] == "True":
            clf.fit(X, y)
            score = Evaluation_Metrics.Cross_validation(clf, X, y)
            Globals.logger.info(score)
            return score, None
        if Globals.config["Evaluation Metrics"]["parameter_search"] == "True":
            n_neighbors = range(3, 11, 2)
            p = range(1, 5)
            leaf_size = range(20, 40)
            param_distributions = dict(n_neighbors=n_neighbors, leaf_size=leaf_size, p=p)
            clf = RandomizedSearchCV(clf, param_distributions, n_iter=100, scoring=None, fit_params=None, n_jobs=-1,
                                     iid='warn', refit=True, cv=10, verbose=2, pre_dispatch='2*n_jobs',
                                     random_state=None, error_score='raise-deprecating', return_train_score='warn')
            best_model = clf.fit(X, y)
            print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
            print('Best C:', best_model.best_estimator_.get_params()['C'])
            return None, None
        else:
            fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
            y_predict = clf.predict(X_test)
            eval_metrics_KNN = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
            return eval_metrics_KNN, clf
    else:
        y_predict = clf.predict(X_test)
        eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
        return eval_metrics_SVM, clf


##### KMeans
def KMeans(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
    Globals.logger.info("Kmeans >>>>>>>")
    if clf is None:
        if Globals.config["Classifiers"]["weighted"] == "True":
            Globals.logger.warn("KMeans does not support weighted classification")
            return

        clf = sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                                     precompute_distances='auto',
                                     verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
        if Globals.config["Evaluation Metrics"]["cross_validate"] == "True":
            score = Evaluation_Metrics.Cross_validation(clf, X, y)
            Globals.logger.info(score)
            return score, None
        if Globals.config["Evaluation Metrics"]["parameter_search"] == "True":
            penalty = ['l1', 'l2']
            C = uniform(loc=0, scale=4)
            solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            param_distributions = {"n_clusters": [int(x) for x in np.linspace(start=5, stop=30, num=1)],
                                   "tol": [0.0001, 0.001, 0.01, 0.1], "max_iter": [300, 500, 700, 1000]}
            clf = RandomizedSearchCV(clf, param_distributions, n_iter=10, scoring=None, fit_params=None, n_jobs=None,
                                     iid='warn', refit=True, cv='warn', verbose=0, pre_dispatch='2*n_jobs',
                                     random_state=None, error_score='raise-deprecating', return_train_score='warn')
            best_model = clf.fit(X, y)
            print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
            print('Best C:', best_model.best_estimator_.get_params()['C'])
            return None, None
        else:
            fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
            y_predict = clf.predict(X_test)
            eval_metrics_kmeans = Evaluation_Metrics.eval_metrics_cluster(y_test, y_predict)
            return eval_metrics_kmeans, clf
    # Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
    else:
        y_predict = clf.predict(X_test)
        eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
        return eval_metrics_SVM, clf


##### Bagging
def Bagging(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
    Globals.logger.info("Bagging >>>>>>>")
    if clf is None:
        if Globals.config["Classifiers"]["weighted"] == "True":
            base_classifier = DecisionTreeClassifier(class_weight='balanced')
        else:
            # base_classifier=DecisionTreeClassifier()
            base_classifier = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=70,
                                                     min_samples_split=2, min_samples_leaf=1,
                                                     min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                                                     max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                     min_impurity_split=None, class_weight=None, presort=False)
        """
        clf=BaggingClassifier(base_estimator=base_classifier, n_estimators=10, max_samples=1.0, max_features=1.0,
            bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None,
            verbose=0)
        """
        clf = BaggingClassifier(base_estimator=base_classifier, n_estimators=90, max_samples=1.0, max_features=1.0,
                                bootstrap=False, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=2,
                                random_state=None,
                                verbose=0)
        if Globals.config["Evaluation Metrics"]["cross_validate"] == "True":
            score = Evaluation_Metrics.Cross_validation(clf, X, y)
            Globals.logger.info(score)
            return score, None
        if Globals.config["Evaluation Metrics"]["parameter_search"] == "True":
            param_distributions = {"n_estimators": [int(x) for x in np.linspace(start=10, stop=100, num=10)],
                                   "max_features": [1.0, 10, 50, 100], "max_samples": [0.25, 0.5, 0.75, 1.0],
                                   "bootstrap": [True, False], "bootstrap_features": [True, False]}
            clf = RandomizedSearchCV(clf, param_distributions, n_iter=20, scoring='f1', fit_params=None, n_jobs=None,
                                     iid='warn', refit=True, cv=10, verbose=2, pre_dispatch='2*n_jobs',
                                     random_state=None, error_score='raise-deprecating', return_train_score='warn')
            best_model = clf.fit(X, y)
            print('Best n_estimators:', best_model.best_estimator_.get_params()['n_estimators'])
            print('Best max_features:', best_model.best_estimator_.get_params()['max_features'])
            print('Best bootstrap:', best_model.best_estimator_.get_params()['bootstrap'])
            print('Best bootstrap_features:', best_model.best_estimator_.get_params()['bootstrap_features'])
            return None, None
        else:
            fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
            y_predict = clf.predict(X_test)
            eval_metrics_bagging = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
            return eval_metrics_bagging, clf
    else:
        y_predict = clf.predict(X_test)
        eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
        return eval_metrics_SVM, clf


#### Boosting
def Boosting(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
    Globals.logger.info("Boosting >>>>>>>")
    if clf is None:
        if Globals.config["Classifiers"]["weighted"] == "True":
            base_classifier = DecisionTreeClassifier(class_weight='balanced')
        else:
            base_classifier = DecisionTreeClassifier()
            base_classifier = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=70,
                                                     min_samples_split=2, min_samples_leaf=1,
                                                     min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                                                     max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                     min_impurity_split=None, class_weight=None, presort=False)

        # clf = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R',
        clf = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=100, learning_rate=1.5, algorithm='SAMME',
                                 random_state=None)
        if Globals.config["Evaluation Metrics"]["cross_validate"] == "True":
            score = Evaluation_Metrics.Cross_validation(clf, X, y)
            Globals.logger.info(score)
            return score, None
        if Globals.config["Evaluation Metrics"]["parameter_search"] == "True":
            param_distributions = {"n_estimators": [int(x) for x in np.linspace(start=10, stop=100, num=10)],
                                   "learning_rate": [0.01, 0.1, 0.25, 1.0, 1.25, 1.5],
                                   "algorithm": ['SAMME', 'SAMME.R']}
            clf = RandomizedSearchCV(clf, param_distributions, n_iter=20, scoring='f1', fit_params=None, n_jobs=-1,
                                     iid='warn', refit=True, cv=10, verbose=2, pre_dispatch='2*n_jobs',
                                     random_state=None, error_score='raise-deprecating', return_train_score='warn')
            best_model = clf.fit(X, y)
            print('Best n_estimators:', best_model.best_estimator_.get_params()['n_estimators'])
            print('Best learning_rate:', best_model.best_estimator_.get_params()['learning_rate'])
            print('Best algorithm:', best_model.best_estimator_.get_params()['algorithm'])
            return None, None
        else:
            fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
            y_predict = clf.predict(X_test)
            eval_metrics_boosting = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
            return eval_metrics_boosting, clf
    else:
        y_predict = clf.predict(X_test)
        eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
        return eval_metrics_SVM, clf


############### imbalanced learning
def DNN(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None):
    if Globals.config["Classifiers"]["weighted"] == "True":
        Globals.logger.warn("DNN does not support weighted classification")
        return
    from sklearn.model_selection import StratifiedKFold
    np.set_printoptions(threshold=np.nan)

    def model_build(dim):
        Globals.logger.debug("Start Building DNN Model >>>>>>")
        K.set_learning_phase(1)  # set learning phase
        model_dnn = Sequential()
        model_dnn.add(Dense(80, kernel_initializer='normal', activation='relu',
                            input_dim=dim))  # units in Dense layer as same as the input dim
        model_dnn.add(Dense(1, activation='sigmoid'))
        model_dnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        Globals.logger.debug("model compile end >>>>>>")
        return model_dnn

    dim = X.shape[1]
    # Globals.logger.info(X[0].transpose().shape)
    model_dnn = model_build(dim)
    if Globals.config["Evaluation Metrics"]["cross_validate"] == "True":
        # return -1
        seed = 7
        np.random.seed(seed)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        cvscores = []
        for train_index, test_index in kfold.split(X, y):
            y_np_array = np.array(y)
            y_train = y_np_array[train_index]
            y_test = y_np_array[test_index]
            model_dnn.fit(X[train_index], y_train, epochs=150, batch_size=10, verbose=0)  # fit the model
            scores = model_dnn.evaluate(X[test_index], y_test, verbose=0)  # evaluate the model
            cvscores.append(scores[1])
        return np.mean(cvscores)

    else:
        model_dnn.fit(X, y, epochs=150, batch_size=100, verbose=0)
        y_predict = model_dnn.predict(X_test)
        eval_metrics_DNN = Evaluation_Metrics.eval_metrics(model_dnn, X, y, y_test, y_predict.round())
        return eval_metrics_DNN


def HDDT():
    # java -cp <path to weka-hddt.jar> weka.classifiers.trees.HTree -U -A -B -t <training file> -T <testing file>

    weka_hddt_path = "weka-hddt-3-7-1.jar"
    subprocess.call(
        ['java', '-cp', weka_hhdt_path, 'weka.classifiers.trees.HTree', '-U', '-A' '-B' '-t', y_predict, y_test])


##To-Do: Add DNN and OLL
####
def rank_classifier(eval_clf_dict, metric_str):
    """

    """
    dict_metric_str = {}
    sorted_eval_clf_dict = {}
    # create the dictionary with the metric
    for clf, val in eval_clf_dict.items():
        # print (clf)
        # print (val[metric_str])
        try:
            dict_metric_str[clf] = val[metric_str]
        except KeyError:
            Globals.logger.warning("Does not work for classifier: {}".format(clf))
    Globals.logger.info("The ranked classifiers on the metric: {}".format(metric_str))
    sorted_dict_metric_str = sorted(((value, key) for (key, value) in dict_metric_str.items()), reverse=True)
    # print (sorted_dict_metric_str)
    for tuple in sorted_dict_metric_str:
        sorted_eval_clf_dict[tuple[1]] = eval_clf_dict[tuple[1]]
    Globals.logger.info(sorted_eval_clf_dict)


def classifiers(X, y, X_test, y_test, X_train_balanced=None, y_train_balanced=None):
    Globals.logger.info("##### Classifiers #####")
    Globals.summary.write("\n##############\n\nClassifiers Used:\n")
    eval_metrics_per_classifier_dict = {}
    if Globals.config["Classification"]["load model"] != "True":
        if X_test is None and Globals.config["Evaluation Metrics"]["cross_validate"] != "True":
            X, X_test, y, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=1)
        if Globals.config["Evaluation Metrics"]["cross_validate"] != "True":
            if Globals.config["Imbalanced Datasets"]["make_imbalanced_dataset"] == "True":
                X_train_balanced, y_train_balanced = Imbalanced_Dataset.Make_Imbalanced_Dataset(X, y)
    trained_model = None
    if not os.path.exists("Data_Dump/Models"):
        os.makedirs("Data_Dump/Models")
    if Globals.config["Extraction"]["BootStrapping"] != "False":
        resampling = int(Globals.config["Extraction"]["BootStrapping"])
    else:
        resampling = 1
    random_state = RandomState(seed=0)
    for iteration in range(resampling):
        # if iteration < 607:
        #	random_state.randint(0, X_test.shape[0], size=(X_test.shape[0],))
        #	continue
        if Globals.config["Extraction"]["BootStrapping"] != "False":
            X_test_i, y_test_i = resample(X_test, y_test, random_state=random_state)
        else:
            X_test_i, y_test_i = X_test, y_test
        run_classifier(X, y, X_test_i, y_test_i, X_train_balanced, y_train_balanced, trained_model,
                       eval_metrics_per_classifier_dict)
    Globals.logger.info(eval_metrics_per_classifier_dict)
    if Globals.config["Classification"]["Rank Classifiers"] == "True":
        rank_classifier(eval_metrics_per_classifier_dict, Globals.config["Classification"]["rank on metric"])


def run_classifier(X, y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model,
                   eval_metrics_per_classifier_dict):
    if Globals.config["Classifiers"]["SVM"] == "True":
        if Globals.config["Classification"]["load model"] == "True":
            trained_model = joblib.load("Data_Dump/Models/model_svm.pkl")
        eval_SVM, model = SVM(X, y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
        eval_metrics_per_classifier_dict['SVM'] = eval_SVM
        if Globals.config["Classification"]["save model"] == "True" and model is not None:
            joblib.dump(model, "Data_Dump/Models/model_svm.pkl")
        Globals.summary.write("SVM\n")
    if Globals.config["Classifiers"]["RandomForest"] == "True":
        if Globals.config["Classification"]["load model"] == "True":
            trained_model = joblib.load("Data_Dump/Models/model_RF.pkl")
        eval_RF, model = RandomForest(X, y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
        eval_metrics_per_classifier_dict['RF'] = eval_RF
        if Globals.config["Classification"]["save model"] == "True" and model is not None:
            joblib.dump(model, "Data_Dump/Models/model_RF.pkl")
        Globals.summary.write("Random Forest\n")
    if Globals.config["Classifiers"]["DecisionTree"] == "True":
        if Globals.config["Classification"]["load model"] == "True":
            trained_model = joblib.load("Data_Dump/Models/model_DT.pkl")
        eval_DT, model = DecisionTree(X, y, X_test, y_test, None, None, trained_model)
        eval_metrics_per_classifier_dict['Dec_tree'] = eval_DT
        if Globals.config["Classification"]["save model"] == "True" and model is not None:
            joblib.dump(model, "Data_Dump/Models/model_DT.pkl")
        Globals.summary.write("Decision Tree \n")
    if Globals.config["Classifiers"]["GaussianNaiveBayes"] == "True":
        if Globals.config["Classification"]["load model"] == "True":
            trained_model = joblib.load("Data_Dump/Models/model_GNB.pkl")
        eval_NB, model = GaussianNaiveBayes(X, y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
        eval_metrics_per_classifier_dict['GNB'] = eval_NB
        if Globals.config["Classification"]["save model"] == "True" and model is not None:
            joblib.dump(model, "Data_Dump/Models/model_GNB.pkl")
        Globals.summary.write("Gaussian Naive Bayes \n")
    if Globals.config["Classifiers"]["MultinomialNaiveBayes"] == "True":
        if Globals.config["Classification"]["load model"] == "True":
            trained_model = joblib.load("Data_Dump/Models/model_MNB.pkl")
        eval_MNB, model = MultinomialNaiveBayes(X, y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
        eval_metrics_per_classifier_dict['MNB'] = eval_MNB
        if Globals.config["Classification"]["save model"] == "True" and model is not None:
            joblib.dump(model, "Data_Dump/Models/model_MNB.pkl")
        Globals.summary.write("Multinomial Naive Bayes \n")
    if Globals.config["Classifiers"]["LogisticRegression"] == "True":
        if Globals.config["Classification"]["load model"] == "True":
            trained_model = joblib.load("Data_Dump/Models/model_LR.pkl")
        eval_LR, model = LogisticRegression(X, y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
        eval_metrics_per_classifier_dict['LR'] = eval_LR
        if Globals.config["Classification"]["save model"] == "True" and model is not None:
            joblib.dump(model, "Data_Dump/Models/model_LR.pkl")
        Globals.summary.write("Logistic Regression\n")
    if Globals.config["Classifiers"]["ELM"] == "True":
        if Globals.config["Classification"]["load model"] == "True":
            trained_model = joblib.load("Data_Dump/Models/model_ELM.pkl")
        eval_elm, model = ELM(X, y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
        eval_metrics_per_classifier_dict['ELM'] = eval_elm
        if Globals.config["Classification"]["save model"] == "True" and model is not None:
            joblib.dump(model, "Data_Dump/Models/model_ELM.pkl")
        Globals.summary.write("ELM\n")
    if Globals.config["Classifiers"]["kNearestNeighbor"] == "True":
        if Globals.config["Classification"]["load model"] == "True":
            trained_model = joblib.load("Data_Dump/Models/model_KNN.pkl")
        eval_knn, model = kNearestNeighbor(X, y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
        eval_metrics_per_classifier_dict['KNN'] = eval_knn
        if Globals.config["Classification"]["save model"] == "True" and model is not None:
            joblib.dump(model, "Data_Dump/Models/model_KNN.pkl")
        Globals.summary.write("kNearest Neighbor\n")
    if Globals.config["Classifiers"]["KMeans"] == "True":
        if Globals.config["Classification"]["load model"] == "True":
            trained_model = joblib.load("Data_Dump/Models/model_Kmeans.pkl")
        eval_kmeans, model = KMeans(X, y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
        eval_metrics_per_classifier_dict['KMeans'] = eval_kmeans
        if Globals.config["Classification"]["save model"] == "True" and model is not None:
            joblib.dump(model, "Data_Dump/Models/model_Kmeans.pkl")
        Globals.summary.write("kMeans \n")
    if Globals.config["Classifiers"]["Bagging"] == "True":
        if Globals.config["Classification"]["load model"] == "True":
            trained_model = joblib.load("Data_Dump/Models/model_bagging.pkl")
        eval_bagging, model = Bagging(X, y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
        eval_metrics_per_classifier_dict['Bagging'] = eval_bagging
        if Globals.config["Classification"]["save model"] == "True" and model is not None:
            joblib.dump(model, "Data_Dump/Models/model_bagging.pkl")
        Globals.summary.write("Bagging \n")
    if Globals.config["Classifiers"]["Boosting"] == "True":
        if Globals.config["Classification"]["load model"] == "True":
            trained_model = joblib.load("Data_Dump/Models/model_boosting.pkl")
        eval_boosting, model = Boosting(X, y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
        eval_metrics_per_classifier_dict['Boosting'] = eval_boosting
        if Globals.config["Classification"]["save model"] == "True" and model is not None:
            joblib.dump(model, "Data_Dump/Models/model_boosting.pkl")
        Globals.summary.write("Boosting \n")
    if Globals.config["Classifiers"]["DNN"] == "True":
        if Globals.config["Classification"]["load model"] == "True":
            trained_model = joblib.load("Data_Dump/Models/model_DNN.pkl")
        eval_dnn = DNN(X, y, X_test, y_test, X_train_balanced, y_train_balanced)
        eval_metrics_per_classifier_dict['DNN'] = eval_dnn
        Globals.summary.write("DNN \n")


def fit_MNB(X, y):
    mnb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    mnb.fit(X, y)
    Globals.logger.info("MNB >>>>>>>")
    joblib.dump(mnb, "Data_Dump/Emails_Training/MNB_model.pkl")
