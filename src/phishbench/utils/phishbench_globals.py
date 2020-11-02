"""
This module contains the global configuration for PhishBench
"""
import argparse
import configparser
import logging
import os
from io import TextIOBase
from typing import Optional

# pylint: disable=global-statement
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

args = None
config = configparser.ConfigParser()
logger: logging.Logger = logging.getLogger('root')
summary: Optional[TextIOBase] = None
output_dir = ""


def parse_args():
    """
    Sets up the argument parser
    """
    global args
    global output_dir
    parser = argparse.ArgumentParser(description='PhishBench Basic Experiment Script')
    parser.add_argument("--version", help="Display the PhishBench version number and exit", action="store_true")
    parser.add_argument("-f", "--config_file", help="The config file to use", type=str, default='Default_Config.ini')
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("-o", "--output_input_dir", help="Output/input directory",
                        type=str, default="PhishBench Output")
    parser.add_argument("-c", "--ignore_confirmation", help="Do not wait for user's confirmation", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_input_dir


def setup_logger(filename='phishbench.log'):
    """
    Sets up the logger
    Parameters
    ----------
    filename: str
        The path of the file to store the log in
    """
    global logger
    logging.captureWarnings(True)

    logger = logging.getLogger('root')
    formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                                  '%m-%d %H:%M:%S')

    if args and args.verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.CRITICAL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if args and args.verbose:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def initialize(config_file):
    """
    Initialize PhishBench with a configuration file.
    Parameters
    ----------
    config_file: str
        The path of the configuration file to initialize PhishBench with
    """
    global args
    global config
    global summary

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    config.read(config_file)
    summary_path = os.path.join(output_dir, config["Summary"]["Path"])
    summary = open(summary_path, 'w')
    setup_logger()


def destroy_globals():
    global summary
    if summary:
        summary.close()
    summary = None
