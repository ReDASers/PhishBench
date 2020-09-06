"""
Arbitrary mock objects
"""

import os
import pathlib
import pickle


def get_mock_object(filename):
    """
    Loads a pickled resource object from file.
    Parameters
    ----------
    filename:
        The filename of the object, with or without the .pkl extension

    Returns
    -------
        The pickled object
    """
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'
    current_file_folder = pathlib.Path(__file__).parent.absolute()
    test_file = os.path.join(current_file_folder, 'resources', 'mock_objects', filename)
    with open(test_file, 'rb') as f:
        result = pickle.load(f)
    return result
