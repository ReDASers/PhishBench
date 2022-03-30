# Change Log

# v2.0.2

## Major changes

* `phishbenh.input.email_input.read_email_from_file` now returns an `EmailMessage` instead of a raw `Message` object.

## Minor Changes

* Updated Dependencies
    * Bumped tldextract version to 3.1.1
    * Changed dnspython version to max-version pin
    * Bumped sklearn to 0.24.2
    * Bumped nltk to >=3.6
    * Bumped textstat to 0.7.2
    * Bumped tldextract to 3.1.2
    * Bumped BeautifulSoup to >=4.10.0
    * Relaxed Scipy to scipy>=1.4.1
    * Bumped TensorFlow to 2.6.3
    * Bumped lxml to 4.6.5
* Added type checks to `feature_extraction.url.extract_features_from_single`
* Added type checks to `feature_extraction.email.extract_features_from_single`
* Renamed `num_unique_attachment` to `num_attachment`
* Optimized feature cleaning performance

## Bug Fixes
* Fixed bug calculating standard deviation in `link_alexa_global_rank`
* PhishBench no longer displays dataset balancing and feature selection options if preprocessing is disabled.
* Fixed bug reading email test set.
* Added missing `rnn_features` import.

# v2.0.1

## Major changes 

* Added features from *Visualizing and Interpreting RNN Models in URL-based Phishing Detection*

## Minor changes

* Updated Dependencies
    * Bumped Tensorflow version to 2.4.0
    * Removed DocumentFeatureSelection as a dependency 
    * Bumped tqdm version to 4.55.0
    * Bumped xgboost version to 1.3.1
    * Bumped sklearn version to 0.24.0
    * Bumped tldextract version to 3.1.0
    * Bumped chardet version to 4.0.0
    * Bumped requests to 2.25.1
* Added input validation for `phishbench.initialize` 

    
## Bug Fixes

* Fixed crash due to single-class sample
* Fixed bug decoding certain HTML emails
* Fixed Issue #291

# v2.0.0

## Major changes
* Added support for splitting dataset into train and test set
* Refactored dataset balancing methods into the `phishbench.feature_preprocessing.balancing` package.
* Moved feature preprocessing script loop to `phishbench.feature_preprocessing`
* Removed legacy code

## Minor changes
* Changed default output folder to `PhishBench Output` 
* Moved log and summary to output folder
* Updated user interaction
* Switched base classifier for RFE from `LinearSVC` to `RandomForestClassifier`
* Added `feature_enabled` to `feature_extraction.settings`


## Bug Fixes 
* Fixed Issue #213

# v1.3.0

## Major changes 
* Reflection Extraction v2
   * Features can now be implemented as a class with a `fit` and `extract` function
   * Fixed default values for multi-valued features
* Added `feature_preproccessing.Vectorizer`
   * Supports `numpy.ndarray` and `scipy.sparse` valued features 
* Moved `dataset.settings` to `input.settings`
* Updated Configuration Files
    * Module toggles moved to `phishbench` section
    * Changed `Email or URL` to `Mode` setting 
* Added `input.read_train_set` and `input.read_test_set` functions
* Removed legacy `Extract_Feature_URL_Testing` and `Extract_Features_URL_Training` functions
    * These functions just wrapped `extract_labeled_dataset`
* Standardized `feature_extraction.email` and `feature_extraction.url` API

## Minor changes 
* Updated user interaction
* Renamed `x_original_authentication_results`, `received_spf`, and `dkim_signature` to `has_{}` for clarity
* Suppressed Warnings from TensorFlow

## Bug fixes
* Fixed non-breaking bug in `export_features_to_csv`

# v1.2.1
## Major changes
* Renamed `extract_url_features` `extract_features_from_list_urls` in `feature_extraction.url`
* Added `phishbench.preproccessing` module to replace `FeatureSupport`
* Added type stub for metrics

## Minor Changes
* Refactored remaining URL/Website features using reflection
* Made package name lowercase
* Improved error message when PhishBench fails to load email

# v1.2.0

## Major Changes
* Refactored the input module
    * Moved `read_email_from_file` to `email_input` submodule
    * Moved `read_url_dataset` to `email_input` submodule
    * Moved `read_email_dataset` to `url_input` submodule
* Removed the dependency on Selenium and Google Chrome
* `URLData` model flattened
    * The downloaded html is now stored in`downloaded_website` instead of `downloaded_website.html` 
    * The final URL is now stored in `final_url` instead of `downloaded_website.final_url`
    * The HTTP headers are now stored in `website_headers` instead of `downloaded_website.headers`
    * Removed `downloaded_website.log`
    
## Minor Changes
* Refactored raw url features using reflection
    
## Bug Fixes
* Fixed looking up whois information with sub-domain urls
* Fixed parsing urls without a scheme

# v1.1.4

## Major Changes

* `url_char_distance` renamed to `url_char_dist`
    * This feature was originally incorrectly named, as it was a distribution, not distance.

## Minor Changes

* Converted URL features to reflection
* Parallelized Random Forest Classifier

## Bug Fixes

* Fixed bug where metric reflection would fail in the presence of custom features. 
* Fixed `Failed to convert a NumPy array to a Tensor` error in `FeedForwardNetwork` with `int` datatypes

# v1.1.3

## Major Changes

* Some URL features renamed for clarity
  * `kolmogorov_shmirnov` -> `char_dist_kolmogorov_shmirnov`
  * `kullback_leibler_divergence` -> `char_dist_kl_divergence`
  * `english_frequency_distance` -> `char_dist_euclidian_distance`
* Changed `URLData` model to store  `URLparse` object instead of individual parts

## Minor Changes

* Converted URL features to reflection 
