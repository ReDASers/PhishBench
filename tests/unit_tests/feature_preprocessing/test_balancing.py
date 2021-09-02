"""
Tests Balancing functions
"""
import unittest
from sklearn.datasets import load_breast_cancer

import numpy as np

from phishbench.feature_preprocessing.balancing import _methods as methods


# pylint: disable=missing-function-docstring
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pylint: disable=too-many-public-methods

class TestBalancing(unittest.TestCase):
    """
    Tests `phishbench.feature_preproccessing.balancing._methods`
    """

    def test_condensed_nearest_neighbor(self):
        data = load_breast_cancer()
        features, labels = methods.condensed_nearest_neighbor(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_edited_nearest_neighbor(self):
        data = load_breast_cancer()
        features, labels = methods.edited_nearest_neighbor(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_repeated_edited_nearest_neighbor(self):
        data = load_breast_cancer()
        features, labels = methods.repeated_edited_nearest_neighbor(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_all_knn(self):
        data = load_breast_cancer()
        features, labels = methods.all_knn(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_instance_hardness_threshold(self):
        data = load_breast_cancer()
        features, labels = methods.instance_hardness_threshold(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_near_miss(self):
        data = load_breast_cancer()
        features, labels = methods.near_miss(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_neighborhood_cleaning_rule(self):
        data = load_breast_cancer()
        features, labels = methods.neighborhood_cleaning_rule(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_one_sided_selection(self):
        data = load_breast_cancer()
        features, labels = methods.one_sided_selection(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_random_undersampling(self):
        data = load_breast_cancer()
        features, labels = methods.random_undersampling(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_tomek_links(self):
        data = load_breast_cancer()
        features, labels = methods.tomek_links(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_adasyn(self):
        data = load_breast_cancer()
        features, labels = methods.adasyn(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_random_oversampling(self):
        data = load_breast_cancer()
        features, labels = methods.random_oversampling(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_smote(self):
        data = load_breast_cancer()
        features, labels = methods.smote(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_borderline_smote(self):
        data = load_breast_cancer()
        features, labels = methods.borderline_smote(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)

    def test_smote_enn(self):
        data = load_breast_cancer()
        features, labels = methods.smote_enn(data.data, data.target)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
