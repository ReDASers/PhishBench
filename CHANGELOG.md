# Change Log

# v1.1.4

## Bug Fixes

* Fixed bug where metric reflection would fail in the presence of custom features. 

# v1.1.3

## Major Changes

* Some URL features renamed for clarity
  * `kolmogorov_shmirnov` -> `char_dist_kolmogorov_shmirnov`
  * `kullback_leibler_divergence` -> `char_dist_kl_divergence`
  * `english_frequency_distance` -> `char_dist_euclidian_distance`
* Changed `URLData` model to store  `URLparse` object instead of individual parts

## Minor Changes

* Converted URL features to reflection 