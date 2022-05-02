The PhishBench platform extracts features from emails and URLs, run classifiers, and evaluates them using evaluation metrics.

![Unit Test](https://github.com/ReDASers/Phishing-Detection/workflows/Unit%20Test/badge.svg)
![URL Integration Test](https://github.com/ReDASers/Phishing-Detection/workflows/URL%20Integration%20Test/badge.svg)
![Email Integration Test](https://github.com/ReDASers/Phishing-Detection/workflows/Email%20Integration%20Test/badge.svg)
![Website TF-IDF Integration Test](https://github.com/ReDASers/Phishing-Detection/workflows/Website%20TF-IDF%20Integration%20Test/badge.svg)
![Documentation Status](https://readthedocs.org/projects/phishbench/badge/?version=latest)


# Installation

PhishBench works on Python 3.7 and 3.8.

To install the latest version on `master`, run

    pip install git+https://github.com/ReDASers/Phishing-Detection.git

To install a specific version, run 

    pip install git+https://github.com/ReDASers/Phishing-Detection.git@{ version number }

For an example, to install version 1.1.4, run 

    pip install git+https://github.com/ReDASers/Phishing-Detection.git@1.1.4

# How to run a PhishBench Basic Experiment 

The PhishBench Basic experiment script performs an experiment with the following workflow:

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

This section contains the highest-level settings for the basic experiment and toggles for each part of the pipeline. 

* The `mode` setting specifies what type of data PhishBench will be operating with. The options are `URL` or `Email`. 
* The `feature extraction` setting toggles feature extraction from the dataset
* The `preprocessing` setting toggles pre-proccesses the features
* The `classification` setting toggles training and evaluation of classifiers. 

 

#### The `Dataset Path` Sections
This section contains the paths of the dataset to be used. You can specify a path using either a relative path to the current directory or an absolute path. 

* In **URL** mode, the subsets can either be a text file or folder of text files with one URL per line.  

* In **Email** mode, the subsets should be folders of eamils, with one file per email. 

#### The `Extraction` Section
This section controls the behavior of the input and feature extraction modules. 

* The `training dataset` setting controls the Basic Experiment Script's training set. If `True`, then PhishBench extracts features from the raw dataset at `path_legit_train` and `path_phish_train`. Otherwise, it will attempt to load a pre-extracted dataset from `OUTPUT_INPUT_DIR`.

* The `testing dataset` setting controls the Basic Experiment Script's testing set. If `True`, then PhishBench extracts features from the raw datset at `path_legit_test` and `path_phish_test`. Otherwise, its behavior will be determined by the `split dataset` setting. 

* If the `split dataset` setting is `True`, then PhishBench will randomly split the training set into a 75/25 train-test split. 

#### The `Features Export` Section

This sections specifies the formats PhishBench will output the extracted features in. Currently, only `csv` is supported.

#### The `Preprocessing` Section

This section contains toggles for the preprocessing pipeline steps.

#### The `Feature Selection` Section

This section contains settings for feature selection. 

* The `number of best features` setting is the number of features to select. 

* The `with tfidf` setting specifies whether to select Tf-IDF features. 

#### The `Feature Selection Methods` Section

This section contains toggles for the feature selection methods.

#### The `Dataset Balancing` Section

This section contains toggles for the dataset balancing methods

#### The `Classification` Section

This section controls the behavior of the classification module. The internal logic is as follows: 

```python
if classification_settings.load_models():
    classifier.load_model()
elif classification_settings.weighted_training():
    classifier.fit_weighted(x_train, y_train)
elif classification_settings.param_search():
    classifier.param_search(x_train, y_train)
else:
    classifier.fit(x_train, y_train)
```

#### The `Classifiers` Section

This section contains toggles for the built-in and user-implemented classifiers.

#### The `Evaluation Metrics` Section

This section contains toggles for the built-in and user-implemented evaluation metrics.

#### The Feature Sections

The rest of the configuration file contains toggles for the built-in and user-implemented features. The `Email_Feature_Types` or `URL_Feature_Types` sections contain toggles for the respective types, and toggles for individual features sectioned by type. 

PhishBench will extract a feature if the following conditions are met: 

1. The feature type matches the mode
2. The feature's type is enabled. 
3. The feature is enabled. 

## Run Experiment

Once you have your configuration file, you can us the `phishbench` command to run the basic experiment. 

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

# Citation

```
@inproceedings{
    10.1145/3372297.3420017,
    author = {Zeng, Victor and Zhou, Xin and Baki, Shahryar and Verma, Rakesh M.},
    title = {PhishBench 2.0: A Versatile and Extendable Benchmarking Framework for Phishing},
    year = {2020},
    isbn = {9781450370899},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3372297.3420017},
    doi = {10.1145/3372297.3420017},
    abstract = {We describe version 2.0 of our benchmarking framework, PhishBench. With the addition of the ability to dynamically load features, metrics, and classifiers, our new and improved framework allows researchers to rapidly evaluate new features and methods for machine-learning based phishing detection. Researchers can compare under identical circumstances their contributions with numerous built-in features, ranking methods, and classifiers used in the literature with the right evaluation metrics. We will demonstrate PhishBench 2.0 and compare it against at least two other automated ML systems.},
    booktitle = {Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security},
    pages = {2077–2079},
    numpages = {3},
    keywords = {machine learning, phishing, automatic framework, benchmarking},
    location = {Virtual Event, USA},
    series = {CCS '20}
}

```
