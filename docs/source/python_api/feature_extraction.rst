``phishbench.feature_extraction``
======================================

.. automodule:: phishbench.feature_extraction
.. currentmodule:: phishbench.feature_extraction


Built-In features
*******************

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

    .. automodule:: phishbench.feature_extraction.url.features.link_tree

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

``number_of_external_content``
###############################

    .. automodule:: phishbench.feature_extraction.url.features.number_of_external_content

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

Email Features
----------------

Header Features
~~~~~~~~~~~~~~~~~

``mime_version``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.mime_version

``header_file_size``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.size_in_bytes

``return_path``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.return_path


``X-mailer``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.x_mailer

``x_originating_hostname``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.x_originating_hostname

``x_originating_ip``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.x_originating_ip

``x_virus_scanned``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.x_virus_scanned

``x_spam_flag``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.x_spam_flag

``received_count``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.received_count

``authentication_results_spf_pass``
##########################################

    .. automodule:: phishbench.feature_extraction.email.features.authentication_results_spf_pass

``authentication_results_dkim_pass``
#############################################

    .. automodule:: phishbench.feature_extraction.email.features.authentication_results_dkim_pass

``has_x_original_authentication_results``
#############################################

    .. automodule:: phishbench.feature_extraction.email.features.has_x_original_authentication_results

``has_received_spf``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.has_received_spf

``has_dkim_signature``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.has_dkim_signature

``compare_sender_domain_message_id_domain``
#############################################

    .. automodule:: phishbench.feature_extraction.email.features.compare_sender_domain_message_id_domain

``compare_sender_return``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.compare_sender_return

``blacklisted_words_subject``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.blacklisted_words_subject

``number_cc``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.number_cc

``number_bcc``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.number_bcc

``number_to``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.number_to

``number_of_words_subject``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.number_of_words_subject

``number_of_characters_subject``
##################################

    .. automodule:: phishbench.feature_extraction.email.features.number_of_characters_subject

``number_of_special_characters_subject``
#########################################

    .. automodule:: phishbench.feature_extraction.email.features.number_of_special_characters_subject

``is_forward``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.fwd_in_subject

``is_reply``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.is_reply

``vocab_richness_subject``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.vocab_richness_subject

Body Features
~~~~~~~~~~~~~~~~~

``is_html``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.is_html

``num_content_type``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type

``num_unique_content_type``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.num_unique_content_type

``num_content_type_text_plain``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_text_plain

``num_content_type_text_html``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_text_html

``num_content_type_multipart_mixed``
############################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_multipart_mixed

``num_content_type_multipart_encrypted``
############################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_multipart_encrypted

``num_content_type_form_data``
###############################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_multipart_form_data

``num_content_type_multipart_byterange``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_multipart_byterange

``num_content_type_multipart_parallel``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_multipart_parallel

``num_content_type_multipart_report``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_multipart_report

``num_content_type_multipart_alternative``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_multipart_alternative

``num_content_type_multipart_signed``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_multipart_signed

``num_content_type_multipart_x_mix_replaced``
####################################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_multipart_x_mixed_replaced

``num_content_disposition``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_disposition

``num_unique_content_disposition``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_unique_content_disposition

``num_charset``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_charset

``num_charset_utf7``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_charset_utf_7

``num_charset_utf8``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_charset_utf_8

``num_charset_gb2312``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_type_charset_gb2312

``num_charset_shift_js``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.email_header_num_content_type_charset_shift_jis

``num_charset_koi``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.email_header_num_content_type_charset_koi

``num_unique_attachment``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_attachment

``num_unique_attachment_filetypes``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_unique_attachment_filetypes

``num_content_transfer_encoding``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_transfer_encoding

``num_unique_content_transfer_encoding``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_unique_content_transfer_encoding

``num_content_transfer_encoding_7bit``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_transfer_encoding_7bit

``num_content_transfer_encoding_8bit``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_transfer_encoding_8bit

``num_content_transfer_encoding_binary``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_transfer_encoding_binary

``num_content_transfer_encoding_quoted_printable``
#######################################################

    .. automodule:: phishbench.feature_extraction.email.features.num_content_transfer_encoding_quoted_printable

``num_words_body``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_words

``num_unique_words_in_body``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.number_of_unique_words_body

``number_of_characters_body``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.number_of_characters_body

``number_unique_chars_body``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.number_unique_chars_body

``number_of_special_characters_body``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.number_of_special_characters_body

``vocab_richness_body``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.vocab_richness_body

``greetings_body``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.greetings_body

``hidden_text``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.hidden_text

``num_href_tag``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_href_tag

``num_end_tag``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_end_tag

``num_open_tag``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_open_tag

``num_on_mouse_over``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.num_on_mouse_over

``blacklisted_words_body``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.blacklisted_words_body

``number_of_scripts``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.number_scripts

``number_of_img_links``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.number_img_links

``function_words_count``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.function_words_counts

``flesh_read_score``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.flesh_read_score

``smog_index``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.smog_index

``flesh_kincaid_score``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.flesh_kincaid_score

``coleman_liau_index``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.coleman_liau_index

``automated_readability_index``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.automated_readability_index

``dale_chall_readability_score``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.dale_chall_readability_score

``difficult_words``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.difficult_words

``linsear_score``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.linsear_score

``gunning_fog``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.gunning_fog

``text_standard``
###########################################

    .. automodule:: phishbench.feature_extraction.email.features.text_standard

References
------------

McGrath, D. Kevin, and Minaxi Gupta. (2008) "*Behind Phishing: An Examination of Phisher Modi Operandi*"

Rakesh Verma and Keith Dyer. 2015. On the Character of Phishing URLs: Accurate and Robust Statistical Learning Classifiers. In Proceedings of the 5th ACM Conference on Data and Application Security and Privacy (CODASPY '15). Association for Computing Machinery, New York, NY, USA, 111–122. DOI:https://doi.org/10.1145/2699026.2699115

A. Das, S. Baki, A. El Aassal, R. Verma and A. Dunbar, "SoK: A Comprehensive Reexamination of Phishing Research From the Security Perspective," in *IEEE Communications Surveys & Tutorials*, vol. 22, no. 1, pp. 671-708, Firstquarter 2020, doi: 10.1109/COMST.2019.2957750.