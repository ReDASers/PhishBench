import os
import pathlib
import pickle
from phishbench.input import URLData


def get_mock_urldata(filename) -> URLData:
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'
    current_file_folder = pathlib.Path(__file__).parent.absolute()
    test_file = os.path.join(current_file_folder, 'resources', 'mock_urls', filename)
    with open(test_file, 'rb') as f:
        result = pickle.load(f)
    return result
