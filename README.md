The PhishBench platform extract features from emails and URLs, run classifiers, and returns the results using different evaluation metrics.

# Installation

PhishBench requires Python 3.7 to run. It does not currently work on Python 3.8. 

To install, clone this repo. Then install dependencies by running 

    pip install -r requirements.txt
    pip install .

# How to run PhishBench


## Make a config file
First create a config file by running 
    
    make-phishbench-config

This will create a starter configuration file `Config_file.ini` that dictates the execution of PhishBench.

If you are extracting features from a dataset, you must specify the location of the dataset via either a relative path to the current directory or an absolute path. 

```
[Dataset Path]
path_legitimate_training = sample_dataset/legit/
path_phishing_training = sample_dataset/phish/
path_legitimate_testing = ../url_2019/legit
path_phishing_testing = ../url_2019/blank
```

Features, Classifiers, Evaluation Metrics, and Imbalanced Methods are toggled via a `True` or `False` like so:

```
Confusion_matrix = True
Cross_validation = False
```

## Run PhishBench
```
usage: phishbench [-h] [-v] [-o OUTPUT_INPUT_DIR] [-c] [-f CONFIG_FILE]

PhishBench

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase output verbosity
  -o OUTPUT_INPUT_DIR, --output_input_dir OUTPUT_INPUT_DIR
                        Output/input directory to read features or dump
                        extracted features
  -c, --ignore_confirmation
                        does not wait or user's confirmation
  -f CONFIG_FILE, --config_file CONFIG_FILE
                        The config file to use.
```



# DESCRIPTION OF MODULES

## Features.py
This module has the definition of all the features that are going to be extracted by the program.
If the user want to add a feature, they should follow the following template: 
```
def feature_name(inputs, list_features, list_time):
    if User_options.feature_name is True:
        start=timeit.timeit()
        code
        list_features["domain_length"]=domain_length
        end=timeit.timeit()
        time=end-start
        list_time["feature_name"]=time
```

if we take the example of a feature that uses 'url' as input then the fucntion will look like this:
```
def url_length(url, list_features, list_time):
    ##global list_features
    if User_options.url_length is True:
        start=timeit.timeit()
        if url=='':
            url_length=0
        else:
            url_length=len(url)
        list_features["url_length"]=url_length
        end=timeit.timeit()
        time=end-start
        list_time["url_length"]=time
```
Notice the test to check if 'url' is empty, then the feature gets 0.
The output of this module are the following:
```
-email_feature_vector_.txt:
-email_features_testing_.txt  (if testing)
-email_features_training_.txt (if training)
-url_feature_vector_.txt
-link_features_testing_.txt (if testing)
-link_features_training_.txt (if training)
```


## Features_Support.py
This module has all the functions that need to run the feature extractions, but are not features per se.
The module is imported into Features.py so any function defined in Features_Support can be called in Feature.py
IMPORTANT: If the user adds a feature in Features.py, then they should also add the following in Feature_Support.py:
```
Features.feature_name(inputs, list_features, list_time)
print("feature_name")
```
This piece of code should be added in one of these different functions in Features_Support.py depending on the nature of the feature: 
```
single_network_features()
single_javascript_features()
single_url_feature()
single_html_features()
single_email_features()
```