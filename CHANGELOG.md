# Change Log
# v1.2.2

## Major changes 
* Renamed `extract_email_features` to `extract_features_list_email` in `feature_extraction.email`
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

## Minor changes 
* Updated user interaction
* Renamed `x_original_authentication_results`, `received_spf`, and `dkim_signature` to `has_{}` for clarity

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
