import argparse
import configparser
import logging

# pylint: disable=global-statement
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

args = None
config = configparser.ConfigParser()
logger: logging.Logger = logging.getLogger('root')
summary = None


def setup_parser():
    global args
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument("--version", help="Display the PhishBench version number", action="store_true")
    parser.add_argument("-f", "--config_file", help="The config file to use.", type=str, default='Default_Config.ini')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-o", "--output_input_dir", help="Output/input directory", type=str, default="Data_Dump")
    parser.add_argument("-c", "--ignore_confirmation", help="does not wait or user's confirmation", action="store_true")
    args = parser.parse_args()

def setup_logger():
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

    file_handler = logging.FileHandler('phishbench.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if args and args.verbose:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def initialize(config_file):
    global args
    global config
    global summary

    config.read(config_file)
    summary = open(config["Summary"]["Path"], 'w')
    setup_logger()


def destroy_globals():
    global summary
    if summary:
        summary.close()
    summary = None
