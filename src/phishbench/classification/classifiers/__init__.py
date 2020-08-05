"""
The `phishbench.classification.classifiers` module contains the built-in classifiers of PhishBench
"""
from .bagging import Bagging
from .boosting import Boosting
from .decision_tree import DecisionTree
from .elm import ExtremeLearningMachine
from .gaussian_nb import GaussianNaiveBayes
from .k_nearest_neighbors import KNN
from .logistic_regression import LogisticRegression
from .multinomial_nb import MultinomialNaiveBayes
from .random_forest import RandomForest
from .svm import SVM
from .xgb import XGBoost
from .feedforward_neural_network import FeedForwardNN
