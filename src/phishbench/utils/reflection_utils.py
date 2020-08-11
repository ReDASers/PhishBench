"""
Functions to help reflection
"""
import importlib
from glob import glob
from os import path


def load_local_modules():
    """
    Loads all python modules in the current working directory
    Returns
    -------
    A list of python modules
    """
    files = glob(r'*.py')
    modules = list()
    for file in files:
        try:
            module_name = "phishbench_loaded_" + path.splitext(file)[0]

            spec = importlib.util.spec_from_file_location(module_name, path.abspath(file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            modules.append(module)
        except ImportError:
            print("Failed to import ", file)
    return modules
