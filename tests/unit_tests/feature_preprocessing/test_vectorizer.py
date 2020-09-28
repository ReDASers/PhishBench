"""
Tests Vectorization functions
"""
import unittest
import numpy as np

from phishbench.feature_preprocessing import Vectorizer
from tests import mock_objects


# pylint: disable=missing-function-docstring
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pylint: disable=too-many-public-methods

class TestVectorizer(unittest.TestCase):
    """
    Tests `phishbench.feature_preproccessing.Vectorizer`
    """
    def test_vectorizer(self):
        features = mock_objects.get_mock_object('vector_test/raw_features')
        expected = mock_objects.get_mock_object('vector_test/vec_features').todense().reshape(174*20)
        vec = Vectorizer()
        x = vec.fit_transform(features).todense().reshape(174*20)
        self.assertTupleEqual(x.shape, expected.shape)
        self.assertTrue(np.all(np.equal(x, expected)))
        
        x = vec.transform(features).todense().reshape(174*20)
        self.assertTupleEqual(x.shape, expected.shape)
        self.assertTrue(np.all(np.equal(x, expected)))

