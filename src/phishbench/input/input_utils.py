import glob
import os
import os.path

from typing import List

from ..utils import phishbench_globals


def enumerate_folder_files(folder_path) -> List[str]:
    """
    Recursively searches a folder for .txt files
    Parameters
    ----------
    folder_path : str
        The path to the folder to search.
    Returns
    -------
        A list containing the paths to every text enumerate in the directory.
    """
    glob_path = os.path.join(folder_path, "**/*.txt")
    return glob.glob(glob_path, recursive=True)


def remove_duplicates(values: List):
    """
    Removes duplicates from a list
    :param values: The list to remove duplicates from
    :return: A new list without duplicates.
    """
    old_len = len(values)
    clean = list(set(values))
    phishbench_globals.logger.info("Removed %d duplicates", old_len - len(clean))
    return clean
