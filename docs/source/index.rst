.. PhishBench documentation master file, created by
   sphinx-quickstart on Thu Jan 14 15:03:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PhishBench |release|
======================================

PhishBench is a extendable framework for benchmarking phishing detection systems designed to let researchers easily
evaluate new features and classification approaches by comparing their effectiveness against existing methods from the
literature. It offers evaluation metrics and methods suitable for imbalanced datasets and includes a rich variety of
algorithms for detection or classification problems. It includes 250 features gleaned from the phishing
detection literature published since 2010, and allows users to easily define new features for testing.

Citation
==========

If you use PhishBench in your work, please use the following citation in your publications. ::

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

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Usage:

   usage/*

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Python API:

   python_api/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
