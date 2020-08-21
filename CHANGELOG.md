# Change Log

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