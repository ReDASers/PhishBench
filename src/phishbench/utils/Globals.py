import argparse
import configparser
import logging

# pylint: disable=global-statement
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

args = None
config = configparser.ConfigParser()
logger = logging.getLogger('root')  # type: logging.Logger
summary = None


def setup_parser():
    global args
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-o", "--output_input_dir",
                        help="Output/input directory to read features or dump extracted features", type=str,
                        default="Data_Dump")
    parser.add_argument("-c", "--ignore_confirmation", help="does not wait or user's confirmation", action="store_true")
    parser.add_argument("-f", "--config_file", help="The config file to use.", type=str, default='Default_Config.ini')
    args = parser.parse_args()


def setup_logger():
    global logger
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


def setup_globals():
    global args
    global config
    global summary
    setup_parser()
    config.read(args.config_file)
    summary = open(config["Summary"]["Path"], 'w')
    setup_logger()


def destroy_globals():
    summary.close()
