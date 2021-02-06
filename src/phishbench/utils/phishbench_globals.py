"""
This module contains the global configuration for PhishBench
"""
import argparse
import configparser
import logging
import os

from .. import settings

# pylint: disable=global-statement
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

config = configparser.ConfigParser()
logger: logging.Logger = logging.getLogger('root')


def parse_args():
    """
    Sets up the argument parser
    """
    parser = argparse.ArgumentParser(description='PhishBench Basic Experiment Script')
    parser.add_argument("--version", help="Display the PhishBench version number and exit", action="store_true")
    parser.add_argument("-f", "--config_file", help="The config file to use", type=str, default='Default_Config.ini')
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("-o", "--output_input_dir", help="Output/input directory",
                        type=str, default="PhishBench Output")
    parser.add_argument("-c", "--ignore_confirmation", help="Do not wait for user's confirmation", action="store_true")
    return parser.parse_args()


def setup_logger(path, verbose=False):
    """
    Sets up the logger
    Parameters
    ----------
    path: str
        The path of the file to store the log in
    verbose:
        Whether or not to output verbosely
    """
    global logger
    logging.captureWarnings(True)

    logger = logging.getLogger('root')
    formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                                  '%m-%d %H:%M:%S')

    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.CRITICAL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.getLogger('tensorflow').setLevel(logging.FATAL)

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def initialize(config_file, output_dir: str = "PhishBench Output", verbose: bool = False):
    """
    Initialize PhishBench with a configuration file.
    Parameters
    ----------
    config_file: str
        The path of the configuration file to initialize PhishBench with
    output_dir: str
        Where to output to
    verbose:
        Whether or not PhishBench should be in verbose mode
    """
    global config

    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"The config file {config_file} does not exist.")

    settings._output_dir = output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    config.read(config_file)
    log_path = os.path.join(output_dir, 'phishbench.log')
    setup_logger(log_path, verbose=verbose)


def destroy_globals():
    pass
