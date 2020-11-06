"""
Feature selection methods
"""
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

import numpy as np


def rfe(x, y, num_features: int):
    """
    Recursive Feature Extraction
    """
    selection_model = RFE(LinearSVC(), n_features_to_select=num_features, step=0.005)
    selection_model.fit(x, y)
    return selection_model, selection_model.ranking_


def chi_squared(x, y, num_features: int):
    """
    Chi-Squared
    """
    selection_model = SelectKBest(chi2, k=num_features)
    selection_model.fit(x, y)
    return selection_model, selection_model.scores_


def information_gain(x, y, num_features: int):
    """
    Information gain
    """
    selection_model = SelectFromModel(DecisionTreeClassifier(criterion='entropy'), threshold=-np.inf,
                                      max_features=num_features)

    selection_model.fit(x, y)
    return selection_model, selection_model.estimator_.feature_importances_


def gini(x, y, num_features: int):
    """
    Gini Coefficient
    """
    selection_model = SelectFromModel(DecisionTreeClassifier(criterion='gini'), threshold=-np.inf,
                                      max_features=num_features)
    selection_model.fit(x, y)
    return selection_model, selection_model.estimator_.feature_importances_


METHODS = {
    "Recursive Feature Elimination": rfe,
    "Chi-2": chi_squared,
    "Information Gain": information_gain,
    "Gini": gini
}
