The PhishBench platform extract features from emails and URLs, run classifiers, and returns the results using different evaluation metrics.

![Test](https://github.com/ReDASers/Phishing-Detection/workflows/Test/badge.svg)

# Installation

PhishBench requires Python 3.7 to run. It has not been validated to work on Python 3.8. 

To install the latest version, run

    pip install git+https://github.com/ReDASers/Phishing-Detection.git

To install a specific version, run 

    pip install git+https://github.com/ReDASers/Phishing-Detection.git@{ verion number }

for an example, to install version 1.1.4, run 

    pip install git+https://github.com/ReDASers/Phishing-Detection.git@1.1.4

# How to run PhishBench


## Make a config file
First create a config file by running `make-phishbench-config`

```
usage: make-phishbench-config [-h] [-v] [-f CONFIG_FILE]

PhishBench Config Generator

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity
  -f CONFIG_FILE, --config_file CONFIG_FILE
                        The name of the config file to generate.
```


This will create a starter configuration file that dictates the execution of PhishBench. Then edit the config file to specify your settings.

If you are extracting features from a dataset, you must specify the location of the dataset via either a relative path to the current directory or an absolute path. 

```
[Dataset Path]
path_legitimate_training = sample_dataset/legit/
path_phishing_training = sample_dataset/phish/
path_legitimate_testing = ../url_2019/legit
path_phishing_testing = ../url_2019/blank
```

You can toggle features, classifiers, evaluation metrics, via a `True` or `False` like so:

```
Confusion_matrix = True
Cross_validation = False
```

## Run PhishBench Basic Experiment
```
usage: phishbench [-h] [--version] [-f CONFIG_FILE] [-v] [-o OUTPUT_INPUT_DIR]
                  [-c]

PhishBench Basic Experiment Script

optional arguments:
  -h, --help            show this help message and exit
  --version             Display the PhishBench version number and exit
  -f CONFIG_FILE, --config_file CONFIG_FILE
                        The config file to use
  -v, --verbose         Increase output verbosity
  -o OUTPUT_INPUT_DIR, --output_input_dir OUTPUT_INPUT_DIR
                        Output/input directory
  -c, --ignore_confirmation
                        Do not wait for user's confirmation
```

## Use PhishBench as a library
For more advance control, you can use PhishBench as a library. To do so, import `phishbench` and the desired modules in your script and call the `initialize` function.

```python
import phishbench

phishbench.initalize('Config_File.ini')
```
