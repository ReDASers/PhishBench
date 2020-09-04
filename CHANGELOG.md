# Change Log

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