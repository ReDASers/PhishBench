The Basic Experiment Script
======================================

Introduction
------------
The Basic Experiment Script performs an experiment with the following workflow:

1. Load data
2. Extract features
3. Pre-process features
4. Train classifiers on a training set
5. Evaluate the trained classifiers on a the test set

The Basic Experiment Script is controlled by a configuration file. In the configuration file, you can select which sections of the workflow you want to run and the features, classifiers, and metrics to use. 

Quick Start
------------

First, generate a configuration file. 

    make-phishbench-config -f phishbench_config.ini
