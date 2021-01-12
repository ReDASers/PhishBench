The PhishBench platform extracts features from emails and URLs, run classifiers, and evaluates them using evaluation metrics.

![Integration Test](https://github.com/ReDASers/Phishing-Detection/workflows/Integration%20Test/badge.svg)
![Unit Test](https://github.com/ReDASers/Phishing-Detection/workflows/Unit%20Test/badge.svg)

# Installation

PhishBench works on Python 3.7 and 3.8.

To install the latest version on `master`, run

    pip install git+https://github.com/ReDASers/Phishing-Detection.git

To install a specific version, run 

    pip install git+https://github.com/ReDASers/Phishing-Detection.git@{ verion number }

For an example, to install version 1.1.4, run 

    pip install git+https://github.com/ReDASers/Phishing-Detection.git@1.1.4

# How to run a PhishBench Basic Experiment 

The PhishBench Basic experiment script perfoms an experiment with the following workflow:

1. Load the dataset
2. Extract features
3. Pre-process features
4. Train classifiers on a training set
5. Evaluate the trained classifiers on a held-out test set

## Make a config file

The basic experiment is controlled by a configuration file. To create a starter config file, run the `make-phishbench-config` command. 

```
usage: make-phishbench-config [-h] [-v] [-f CONFIG_FILE]

PhishBench Config Generator

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity
  -f CONFIG_FILE, --config_file CONFIG_FILE
                        The name of the config file to generate.
```

Example:

```
make-phishbench-config -f Test.ini
```


### Anatomy of a config file

The PhishBench Configuration File is an `ini` file defined according to the Python [ConfigParser](https://docs.python.org/3/library/configparser.html) specification. In general, most settings are binary features which can be toggled via a `True` or `False` like so:

```
Confusion_matrix = True
Cross_validation = False
```

#### The `PhishBench` Section

This section contains the highest-level settings for the basic experiment. The `mode` parameter specifies what type of data PhishBench will be operating with. The options are `URL` or `Email`. and toggles for each part of the pipeline.

 `feature extraction` extracts features from the dataset, `preprocessing` pre-proccesses the features, and `classification` trains and evaluates classifiers. 

If you are extracting features from a dataset, you must specify the location of the dataset via either a relative path to the current directory or an absolute path. 

#### The `Dataset Path` Section
This section contains the paths of the dataset to be used. If your dataset is not split into a train and test set, then leave the test set blank and elable the `split dataset` option in the `Extraction` section. 

In **URL** mode, the datset can either be a text file or folder of text files with one URL per line.  

In **Email** mode, the datset should be a folder of eamils, with one file per email. 

#### The `Extraction` Section

This section contains the settings for the feature extraction module. 

## Run Experiment
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

# Use PhishBench as a library
For experiments which deviate from the basic experiment workflow, you can also use PhishBench as a library. To do so, import `phishbench` and the desired modules in your script and call the `initialize` function.

```python
import phishbench

phishbench.initalize('Config_File.ini')
```
