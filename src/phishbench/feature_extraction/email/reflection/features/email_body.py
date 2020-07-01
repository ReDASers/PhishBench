import re

from textstat import textstat
from bs4 import BeautifulSoup

from . import helpers
from .. import register_feature, FeatureType
from phishbench.input.email_input import EmailBody


@register_feature(FeatureType.EMAIL_BODY, 'is_html')
def email_body_test_html(body: EmailBody):
    return body.is_html


# region Content Type

@register_feature(FeatureType.EMAIL_BODY, 'num_content_type')
def email_body_num_content_type(body: EmailBody):
    return len(body.content_type_list)


@register_feature(FeatureType.EMAIL_BODY, 'num_unique_content_type')
def email_body_num_unique_content_type(body: EmailBody):
    return len(set(body.charset_list))


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_text_plain')
def email_body_num_content_type_text_plain(body: EmailBody):
    return body.content_type_list.count("text/plain")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_text_html')
def email_body_num_content_type_text_html(body: EmailBody):
    return body.content_type_list.count("text/html")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_mixed')
def email_body_num_content_type_multipart_mixed(body: EmailBody):
    return body.content_type_list.count("multipart/mixed")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_encrypted')
def email_body_num_content_type_multipart_encrypted(body: EmailBody):
    return body.content_type_list.count("multipart/encrypted")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_form_data')
def email_body_num_content_type_multipart_form_data(body: EmailBody):
    return body.content_type_list.count("multipart/form-data")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_byterange')
def email_body_num_content_type_multipart_byterange(body: EmailBody):
    return body.content_type_list.count("multipart/byterange")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_parallel')
def email_body_num_content_type_multipart_parallel(body: EmailBody):
    return body.content_type_list.count("multipart/parallel")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_report')
def email_body_num_content_type_multipart_report(body: EmailBody):
    return body.content_type_list.count("multipart/report")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_alternative')
def email_body_num_content_type_multipart_alternative(body: EmailBody):
    return body.content_type_list.count("multipart/alternative")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_digest')
def email_body_num_content_type_multipart_digest(body: EmailBody):
    return body.content_type_list.count("multipart/digest")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_signed')
def email_body_num_content_type_multipart_signed(body: EmailBody):
    return body.content_type_list.count("multipart/signed")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_x_mix_replaced')
def email_body_num_content_type_multipart_x_mixed_replaced(body: EmailBody):
    return body.content_type_list.count("multipart/x-mixed-replaced")


# endregion

@register_feature(FeatureType.EMAIL_BODY, 'num_unique_content_disposition')
def email_body_num_unique_content_disposition(body: EmailBody):
    return len(set(body.content_disposition_list))


# region Charset


@register_feature(FeatureType.EMAIL_BODY, 'num_charset')
def email_body_num_charset(body: EmailBody):
    return len(body.charset_list)


@register_feature(FeatureType.EMAIL_BODY, 'num_unique_charset')
def email_body_num_unique_charset(body: EmailBody):
    return len(set(body.charset_list))


@register_feature(FeatureType.EMAIL_BODY, 'num_charset_us_ascii')
def email_body_num_content_type_charset_us_ascii(body: EmailBody):
    return body.charset_list.count("us_ascii")


@register_feature(FeatureType.EMAIL_BODY, 'num_charset_utf8')
def email_body_num_content_type_charset_utf_8(body: EmailBody):
    return body.charset_list.count("utf_8")


@register_feature(FeatureType.EMAIL_BODY, 'num_charset_utf7')
def email_body_num_content_type_charset_utf_7(body: EmailBody):
    return body.charset_list.count("utf_7")


@register_feature(FeatureType.EMAIL_BODY, 'num_charset_gb2312')
def email_body_num_content_type_charset_gb2312(body: EmailBody):
    return body.charset_list.count("gb2312")


@register_feature(FeatureType.EMAIL_BODY, 'num_charset_shift_js')
def email_header_num_content_type_charset_shift_jis(body: EmailBody):
    return body.charset_list.count("shift_jis")


@register_feature(FeatureType.EMAIL_BODY, 'num_charset_koi')
def email_header_num_content_type_charset_koi(body: EmailBody):
    return body.charset_list.count("koi")


@register_feature(FeatureType.EMAIL_BODY, 'num_charset_iso2022-jp')
def email_header_num_content_type_charset_iso2022_jp(body: EmailBody):
    return body.charset_list.count("iso2022-jp")


# endregion

# region Attachments


@register_feature(FeatureType.EMAIL_BODY, 'num_unique_attachment')
def email_body_num_attachment(body: EmailBody):
    return body.num_attachment


@register_feature(FeatureType.EMAIL_BODY, 'num_unique_attachment_filetypes')
def email_body_num_unique_attachment_filetypes(body: EmailBody):
    return len(set(body.file_extension_list))


# endregion

# region Content Transfer Encoding

@register_feature(FeatureType.EMAIL_BODY, 'num_content_transfer_encoding')
def email_body_num_content_transfer_encoding(body: EmailBody):
    return len(body.content_transfer_encoding_list)


@register_feature(FeatureType.EMAIL_BODY, 'num_unique_content_transfer_encoding')
def email_body_num_unique_content_transfer_encoding(body: EmailBody):
    return len(set(body.content_transfer_encoding_list))


@register_feature(FeatureType.EMAIL_BODY, 'num_7bit_content_transfer_encoding')
def email_body_num_content_transfer_encoding_7bit(body: EmailBody):
    return body.content_transfer_encoding_list.count('7bit')


@register_feature(FeatureType.EMAIL_BODY, 'num_8bit_content_transfer_encoding')
def email_body_num_content_transfer_encoding_8bit(body: EmailBody):
    return body.content_transfer_encoding_list.count('8bit')


@register_feature(FeatureType.EMAIL_BODY, 'num_binary_content_transfer_encoding')
def email_body_num_content_transfer_encoding_binary(body: EmailBody):
    return body.content_transfer_encoding_list.count('binary')


@register_feature(FeatureType.EMAIL_BODY, 'num_quoted_printable_content_transfer_encoding')
def email_body_num_content_transfer_encoding_quoted_printable(body: EmailBody):
    return body.content_transfer_encoding_list.count('quoted-printable')


# # endregion


@register_feature(FeatureType.EMAIL_BODY, 'difficult_words')
def email_body_difficult_words(body: EmailBody) -> int: \
        return textstat.difficult_words(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'num_words')
def email_body_num_words(body: EmailBody) -> int:
    return len(re.findall(r'\w+', body.text))


@register_feature(FeatureType.EMAIL_BODY, 'num_unique_words_in_body')
def email_body_number_of_unique_words_body(body: EmailBody) -> int:
    return len(set(re.findall(r'\w+', body.text)))


@register_feature(FeatureType.EMAIL_BODY, 'number_of_characters_body')
def email_body_number_of_characters_body(body: EmailBody) -> int:
    return len(re.findall(r'\w', body.text))


@register_feature(FeatureType.EMAIL_BODY, 'number_of_special_characters_body')
def email_body_number_of_special_characters_body(body: EmailBody) -> int:
    if body.text is None:
        return 0
    number_of_characters_body = len(re.findall(r'\w', body.text))
    number_of_spaces = len(re.findall(r' ', body.text))
    return len(body.text) - number_of_characters_body - number_of_spaces


@register_feature(FeatureType.EMAIL_BODY, 'vocab_richness_body')
def email_body_vocab_richness_body(body: EmailBody):
    if body.text is None:
        return 0
    return helpers.yule(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'number_of_html_tags_body')
def email_body_number_of_html_tags_body(body: EmailBody):
    if body.text is None:
        return 0
    return len(re.findall(r'<.*>', body.text))


@register_feature(FeatureType.EMAIL_BODY, 'number_unique_chars_body')
def email_body_number_unique_chars_body(body: EmailBody):
    if body.text is None:
        return 0
    return len(set(body.text)) - 1


@register_feature(FeatureType.EMAIL_BODY, 'greetings_body')
def email_body_greetings_body(body: EmailBody):
    if body.text is None:
        return False
    dear_user = re.compile(r'Dear User', flags=re.IGNORECASE)
    return re.search(dear_user, body.text) is not None


@register_feature(FeatureType.EMAIL_BODY, 'hidden_text')
def email_body_hidden_text(body: EmailBody):
    if body.text is None:
        return False
    regex_font_color = re.compile(r'<font +color="#FFFFF[0-9A-F]"', flags=re.DOTALL)
    return regex_font_color.search(body.text) is not None


@register_feature(FeatureType.EMAIL_BODY, 'count_href_tag')
def email_body_count_href_tag(body: EmailBody) -> int:
    if body.text is None:
        return 0
    ultimate_regexp = re.compile(r"(?i)</?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)/?>",
                                 flags=re.MULTILINE)
    count_href_tag = 0
    for match in re.finditer(ultimate_regexp, body.text):
        if repr(match.group()).startswith("'<a"):
            count_href_tag += 1
    return count_href_tag


@register_feature(FeatureType.EMAIL_BODY, 'end_tag_count')
def email_body_end_tag_count(body: EmailBody) -> int:
    if body.text is None:
        return 0
    ultimate_regexp = re.compile(r"(?i)</?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)/?>",
                                 flags=re.MULTILINE)
    end_tag_count = 0
    for match in re.finditer(ultimate_regexp, body.text):
        if repr(match.group()).startswith("'</"):
            end_tag_count += 1


@register_feature(FeatureType.EMAIL_BODY, 'open_tag_count')
def email_body_open_tag_count(body: EmailBody) -> int:
    if body.text is None:
        return 0
    ultimate_regexp = re.compile(r"(?i)</?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)/?>",
                                 flags=re.MULTILINE)
    open_tag_count = 0
    for match in re.finditer(ultimate_regexp, body.text):
        if not repr(match.group()).startswith("'</"):
            open_tag_count += 1
    return open_tag_count


@register_feature(FeatureType.EMAIL_BODY, 'on_mouse_over')
def email_body_on_mouse_over(body: EmailBody) -> int:
    if body.text is None:
        return 0
    ultimate_regexp = re.compile(r"(?i)</?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)/?>",
                                 flags=re.MULTILINE)
    on_mouse_over = 0
    for match in re.finditer(ultimate_regexp, body.text):
        if repr(match.group()).startswith("'<a onmouseover"):
            on_mouse_over += 1
    return on_mouse_over


@register_feature(FeatureType.EMAIL_BODY, 'blacklisted_words_body')
def email_body_blacklisted_words_body(body: EmailBody):
    body_word_blacklist = ["urgent", "account", "closing", "act now", "click here", "limited", "suspension",
                           "your account", "verify your account", "agree", 'bank', 'dear', "update", "confirm",
                           "customer", "client", "suspend", "restrict", "verify", "login", "ssn", 'username', 'click',
                           'log', 'inconvenient', 'alert', 'paypal']
    blacklist_body_count = {}

    if body.text is None:
        for word in body_word_blacklist:
            blacklist_body_count[word] = 0
        return blacklist_body_count

    for word in body_word_blacklist:
        word_count = len(re.findall(word, body.text, re.IGNORECASE))
        blacklist_body_count[word] = word_count
    return blacklist_body_count


@register_feature(FeatureType.EMAIL_BODY, 'Number_Of_Scripts')
def email_body_number_scripts(body: EmailBody):
    if not body.is_html:
        return 0
    soup = BeautifulSoup(body.text, "html.parser")
    return len(soup.find_all('script'))


@register_feature(FeatureType.EMAIL_BODY, 'Number_Of_Img_Links')
def email_body_number_img_links(body: EmailBody):
    if not body.is_html:
        return 0
    soup = BeautifulSoup(body.html, "html.parser")
    return len(soup.find_all('img'))


@register_feature(FeatureType.EMAIL_BODY, 'Function_Words_Count')
def email_body_function_words_counts(body: EmailBody):
    if body.text is None:
        return 0
    function_words_list = {"a", "about", "above", "after", "again", "against", "ago", "ahead", "all", "almost",
                           "almost", "along", "already", "also", "", "although", "always", "am", "among", "an",
                           "and", "any", "are", "aren't", "around", "as", "at", "away", "backward", "backwards", "be",
                           "because", "before", "behind", "below", "beneath", "beside", "between", "both", "but",
                           "by", "can", "cannot", "can't", "cause", "'cos", "could", "couldn't", "'d", "had", "despite",
                           "did", "didn't", "do", "does", "doesn't", "don't", "down", "during", "each", "either",
                           "even", "ever", "every", "except", "for", "faw", "forward", "from", "frm", "had",
                           "hadn't", "has", "hasn't", "have", "hv", "haven't", "he", "hi", "her", "here", "hers",
                           "herself", "him", "hm", "himself", "his", "how", "however", "I", "if", "in", "inside",
                           "inspite", "instead", "into", "is", "isn't", "it", "its", "itself", "just", "'ll", "will",
                           "shall", "least", "less", "like", "'m", "them", "many", "may", "mayn't", "me", "might",
                           "mightn't", "mine", "more", "most", "much", "must", "mustn't", "my", "myself", "near",
                           "need", "needn't", "needs", "neither", "never", "no", "none", "nor", "not", "now", "of",
                           "off", "often", "on", "once", "only", "onto", "or", "ought", "oughtn't", "our", "ours",
                           "ourselves", "out", "outside", "over", "past", "perhaps", "quite", "'re", "rather", "'s", "",
                           "seldom", "several", "shall", "shan't", "she", "should", "shouldn't", "since", "so", "some",
                           "sometimes", "soon", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
                           "there", "therefore", "these", "they", "this", "those", "though", "", "through", "thus",
                           "till", "to", "together", "too", "towards", "under", "unless", "until", "up", "upon", "us",
                           "used", "usedn't", "usen't", "usually", "'ve", "very", "was", "wasn't", "we", "well", "were",
                           "weren't", "what", "when", "where", "whether", "which", "while", "who", "whom", "whose",
                           "why", "will", "with", "without", "won't", "would", "wouldn't", "yet", "you", "your",
                           "yours", "yourself", "yourselves"}
    function_words_count = 0
    for word in body.text.split(' '):
        if word in function_words_list:
            function_words_count = +1
    return function_words_count


# region texststat metrics
@register_feature(FeatureType.EMAIL_BODY, 'flesh_read_score')
def email_body_flesh_read_score(body: EmailBody):
    return textstat.flesch_reading_ease(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'smog_index')
def email_body_smog_index(body: EmailBody):
    return textstat.smog_index(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'flesh_kincaid_score')
def email_body_flesh_kincaid_score(body: EmailBody):
    return textstat.flesch_kincaid_grade(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'coleman_liau_index')
def email_body_coleman_liau_index(body: EmailBody):
    return textstat.coleman_liau_index(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'automated_readability_index')
def email_body_automated_readability_index(body: EmailBody):
    return textstat.automated_readability_index(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'dale_chall_readability_score')
def email_body_dale_chall_readability_score(body: EmailBody):
    return textstat.dale_chall_readability_score(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'difficult_words')
def email_body_difficult_words(body: EmailBody):
    return textstat.difficult_words(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'linsear_score')
def email_body_linsear_score(body: EmailBody):
    return textstat.linsear_write_formula(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'gunning_fog')
def email_body_gunning_fog(body: EmailBody):
    return textstat.gunning_fog(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'text_standard')
def email_body_text_standard(body: EmailBody):
    return textstat.text_standard(body.text)


# endregion


# region Helper Methods



