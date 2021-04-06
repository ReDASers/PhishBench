``phishbench.feature_extraction``
======================================

.. automodule:: phishbench.feature_extraction
.. currentmodule:: phishbench.feature_extraction


Built-In features
-----------------

URL Features
------------

URL Features
~~~~~~~~~~~~

``average_domain_token_length``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.average_domain_token_length 

``average_path_token_length``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.average_path_token_length  

``brand_in_url``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.brand_in_url 

``char_dist``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.char_dist 

``char_dist_euclidian_distance``
##################################

    .. automodule:: phishbench.feature_extraction.url.features.char_dist_euclidian_distance

``char_dist_kl_divergence``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.char_dist_kl_divergence 

``char_dist_kolmogorov_shmirnov``
##################################

    .. automodule:: phishbench.feature_extraction.url.features.char_dist_kolmogorov_shmirnov  

``consecutive_numbers``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.consecutive_numbers 

``digit_letter_ratio``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.digit_letter_ratio 

``domain_length``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.domain_length 

``domain_letter_occurrence``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.domain_letter_occurrence

``double_slashes_in_path``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.double_slashes_in_path

``has_anchor_tag``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.has_anchor_tag 

``has_at_symbol``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.has_at_symbol  

``has_hex_characters``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.has_hex_characters  

``has_https``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.has_https 

``has_more_than_three_dots``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.has_more_than_three_dots

``has_port``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.has_port  

``has_www_in_middle``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.has_www_in_middle  

``http_in_middle``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.http_in_middle   

``is_common_tld``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.is_common_tld   

``is_ip_addr``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.is_ip_addr  

``is_whitelisted``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.is_whitelisted  

``longest_domain_token_length``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.longest_domain_token_length 

``null_in_domain``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.null_in_domain

``num_punctuation``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.num_punctuation

``number_of_dashes``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_dashes

``number_of_digits``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_digits

``number_of_dots``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_dots

``number_of_slashes``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_slashes

``protocol_port_match``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.protocol_port_match

``special_char_count``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.special_char_count

``special_pattern``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.special_pattern

``token_count``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.token_count

``top_level_domain``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.top_level_domain

``url_length``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.url_length

Network Features
~~~~~~~~~~~~~~~~~~

``as_number``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.as_number

``creation_date``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.creation_date

``dns_ttl``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.dns_ttl

``expiration_date``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.expiration_date

``number_name_server``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_name_server

``updated_date``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.updated_date

HTML Features
~~~~~~~~~~~~~~~~~~

``website_tfidf``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.WebsiteTfidf

``content_length``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.content_length_header

``website_content_type``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.content_type_header

``has_password_input``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.has_password_input

``is_redirect``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.is_redirect

``link_alexa_global_rank``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.link_alexa_global_rank

``link_tree_features``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.link_tree_features

``number_of_anchor``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_anchor

``number_of_audio``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_audio

``number_of_body``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_body

``number_of_embed``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_embed

``number_of_external_links``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_external_links

``number_of_head``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_head

``number_of_hidden_div``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_hidden_div

``number_of_hidden_iframe``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_hidden_iframe

``number_of_hidden_input``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_hidden_input

``number_of_hidden_object``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_hidden_object

``number_of_hidden_svg``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_hidden_svg

``number_of_html``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_html

``number_of_iframe``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_iframe

``number_of_img``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_img

``number_of_internal_content``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_internal_content

``number_of_internal_links``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_internal_links

``number_of_scripts``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_scripts

``number_of_tags``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_tags

``number_of_title``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_title

``number_of_video``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_video

``number_suspicious_content``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_suspicious_content

``x_powered_by``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.x_powered_by_header

Javascript Features
~~~~~~~~~~~~~~~~~~~~

``number_of_escape``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_escape

``number_of_eval``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_eval

``number_of_event_attachment``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_event_attachment

``number_of_event_dispatch``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_event_dispatch

``number_of_exec``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_exec

``number_of_iframes_in_script``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_iframes_in_script

``number_of_link``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_link

``number_of_search``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_search

``number_of_set_timeout``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_set_timeout

``number_of_unescape``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_unescape

``right_click_modified``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.right_click_modified

References
------------

McGrath, D. Kevin, and Minaxi Gupta. (2008) "Behind Phishing: An Examination of Phisher Modi Operandi"

Verma, Rakesh, and Keith Dyer. (2015) "On the Character of Phishing URLs"