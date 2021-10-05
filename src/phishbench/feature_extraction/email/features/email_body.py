"""
This module contains the built-in email body features.
"""
import re

from bs4 import BeautifulSoup
from textstat import textstat

from . import helpers
from ...reflection import register_feature, FeatureType
from ....input.email_input.models import EmailBody


@register_feature(FeatureType.EMAIL_BODY, 'is_html')
def is_html(body: EmailBody):
    """
    Whether or not the email has a HTML body
    """
    return body.is_html


# region Content Type

@register_feature(FeatureType.EMAIL_BODY, 'num_content_type')
def num_content_type(body: EmailBody):
    return len(body.content_type_list)


@register_feature(FeatureType.EMAIL_BODY, 'num_unique_content_type')
def num_unique_content_type(body: EmailBody):
    return len(set(body.content_type_list))


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_text_plain')
def num_content_type_text_plain(body: EmailBody):
    return body.content_type_list.count("text/plain")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_text_html')
def num_content_type_text_html(body: EmailBody):
    return body.content_type_list.count("text/html")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_mixed')
def num_content_type_multipart_mixed(body: EmailBody):
    return body.content_type_list.count("multipart/mixed")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_encrypted')
def num_content_type_multipart_encrypted(body: EmailBody):
    return body.content_type_list.count("multipart/encrypted")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_form_data')
def num_content_type_multipart_form_data(body: EmailBody):
    return body.content_type_list.count("multipart/form-data")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_byterange')
def num_content_type_multipart_byterange(body: EmailBody):
    return body.content_type_list.count("multipart/byterange")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_parallel')
def num_content_type_multipart_parallel(body: EmailBody):
    return body.content_type_list.count("multipart/parallel")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_report')
def num_content_type_multipart_report(body: EmailBody):
    return body.content_type_list.count("multipart/report")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_alternative')
def num_content_type_multipart_alternative(body: EmailBody):
    return body.content_type_list.count("multipart/alternative")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_digest')
def num_content_type_multipart_digest(body: EmailBody):
    return body.content_type_list.count("multipart/digest")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_signed')
def num_content_type_multipart_signed(body: EmailBody):
    return body.content_type_list.count("multipart/signed")


@register_feature(FeatureType.EMAIL_BODY, 'num_content_type_multipart_x_mix_replaced')
def num_content_type_multipart_x_mixed_replaced(body: EmailBody):
    return body.content_type_list.count("multipart/x-mixed-replaced")


# endregion

@register_feature(FeatureType.EMAIL_BODY, 'num_content_disposition')
def num_content_disposition(body: EmailBody):
    return len(body.content_disposition_list)


@register_feature(FeatureType.EMAIL_BODY, 'num_unique_content_disposition')
def num_unique_content_disposition(body: EmailBody):
    return len(set(body.content_disposition_list))


# region Charset


@register_feature(FeatureType.EMAIL_BODY, 'num_charset')
def num_charset(body: EmailBody):
    return len(body.charset_list)


@register_feature(FeatureType.EMAIL_BODY, 'num_unique_charset')
def num_unique_charset(body: EmailBody):
    return len(set(body.charset_list))


@register_feature(FeatureType.EMAIL_BODY, 'num_charset_us_ascii')
def num_content_type_charset_us_ascii(body: EmailBody):
    return body.charset_list.count("us_ascii")


@register_feature(FeatureType.EMAIL_BODY, 'num_charset_utf7')
def num_content_type_charset_utf_7(body: EmailBody):
    return body.charset_list.count("utf_7")


@register_feature(FeatureType.EMAIL_BODY, 'num_charset_utf8')
def num_content_type_charset_utf_8(body: EmailBody):
    return body.charset_list.count("utf_8")


@register_feature(FeatureType.EMAIL_BODY, 'num_charset_gb2312')
def num_content_type_charset_gb2312(body: EmailBody):
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
def num_attachment(body: EmailBody):
    return body.num_attachment


@register_feature(FeatureType.EMAIL_BODY, 'num_unique_attachment_filetypes')
def num_unique_attachment_filetypes(body: EmailBody):
    return len(set(body.file_extension_list))


# endregion

# region Content Transfer Encoding

@register_feature(FeatureType.EMAIL_BODY, 'num_content_transfer_encoding')
def num_content_transfer_encoding(body: EmailBody):
    return len(body.content_transfer_encoding_list)


@register_feature(FeatureType.EMAIL_BODY, 'num_unique_content_transfer_encoding')
def num_unique_content_transfer_encoding(body: EmailBody):
    return len(set(body.content_transfer_encoding_list))


@register_feature(FeatureType.EMAIL_BODY, 'num_content_transfer_encoding_7bit')
def num_content_transfer_encoding_7bit(body: EmailBody):
    return body.content_transfer_encoding_list.count('7bit')


@register_feature(FeatureType.EMAIL_BODY, 'num_content_transfer_encoding_8bit')
def num_content_transfer_encoding_8bit(body: EmailBody):
    return body.content_transfer_encoding_list.count('8bit')


@register_feature(FeatureType.EMAIL_BODY, 'num_content_transfer_encoding_binary')
def num_content_transfer_encoding_binary(body: EmailBody):
    return body.content_transfer_encoding_list.count('binary')


@register_feature(FeatureType.EMAIL_BODY, 'num_content_transfer_encoding_quoted_printable')
def num_content_transfer_encoding_quoted_printable(body: EmailBody):
    return body.content_transfer_encoding_list.count('quoted-printable')


# endregion

@register_feature(FeatureType.EMAIL_BODY, 'num_words_body')
def num_words(body: EmailBody) -> int:
    return len(re.findall(r'\w+', body.text))


@register_feature(FeatureType.EMAIL_BODY, 'num_unique_words_in_body')
def number_of_unique_words_body(body: EmailBody) -> int:
    return len(set(re.findall(r'\w+', body.text)))


@register_feature(FeatureType.EMAIL_BODY, 'number_of_characters_body')
def number_of_characters_body(body: EmailBody) -> int:
    return len(re.findall(r'\w', body.text))


@register_feature(FeatureType.EMAIL_BODY, 'number_of_special_characters_body')
def number_of_special_characters_body(body: EmailBody) -> int:
    if body.text:
        len(re.findall(r'_|[^\w\s]', body.text))
    return 0


@register_feature(FeatureType.EMAIL_BODY, 'number_unique_chars_body')
def number_unique_chars_body(body: EmailBody):
    if body.text is None:
        return 0
    return len(set(body.text)) - 1


@register_feature(FeatureType.EMAIL_BODY, 'vocab_richness_body')
def vocab_richness_body(body: EmailBody):
    if body.text is None:
        return 0
    return helpers.yule(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'number_of_html_tags_body')
def number_of_html_tags_body(body: EmailBody):
    if body.text is None:
        return 0
    return len(re.findall(r'<.*>', body.text))


@register_feature(FeatureType.EMAIL_BODY, 'greetings_body')
def greetings_body(body: EmailBody):
    if body.text is None:
        return False
    dear_user = re.compile(r'Dear User', flags=re.IGNORECASE)
    return re.search(dear_user, body.text) is not None


@register_feature(FeatureType.EMAIL_BODY, 'hidden_text')
def hidden_text(body: EmailBody):
    if body.text is None:
        return False
    regex_font_color = re.compile(r'<font +color="#FFFFF[0-9A-F]"', flags=re.DOTALL)
    return regex_font_color.search(body.text) is not None


@register_feature(FeatureType.EMAIL_BODY, 'num_href_tag')
def num_href_tag(body: EmailBody) -> int:
    if body.text is None:
        return 0
    ultimate_regexp = re.compile(r"(?i)</?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)/?>",
                                 flags=re.MULTILINE)
    count = 0
    for match in re.finditer(ultimate_regexp, body.text):
        if repr(match.group()).startswith("'<a"):
            count += 1
    return count


@register_feature(FeatureType.EMAIL_BODY, 'num_end_tag')
def num_end_tag(body: EmailBody) -> int:
    if body.text is None:
        return 0
    ultimate_regexp = re.compile(r"(?i)</?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)/?>",
                                 flags=re.MULTILINE)
    count = 0
    for match in re.finditer(ultimate_regexp, body.text):
        if repr(match.group()).startswith("'</"):
            count += 1
    return count


@register_feature(FeatureType.EMAIL_BODY, 'num_open_tag')
def num_open_tag(body: EmailBody) -> int:
    if body.text is None:
        return 0
    ultimate_regexp = re.compile(r"(?i)</?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)/?>",
                                 flags=re.MULTILINE)
    count = 0
    for match in re.finditer(ultimate_regexp, body.text):
        if not repr(match.group()).startswith("'</"):
            count += 1
    return count


@register_feature(FeatureType.EMAIL_BODY, 'num_on_mouse_over')
def num_on_mouse_over(body: EmailBody) -> int:
    if body.text is None:
        return 0
    ultimate_regexp = re.compile(r"(?i)</?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)/?>",
                                 flags=re.MULTILINE)
    count = 0
    for match in re.finditer(ultimate_regexp, body.text):
        if repr(match.group()).startswith("'<a onmouseover"):
            count += 1
    return count


BODY_WORD_BLACKLIST = ["urgent", "account", "closing", "act now", "click here", "limited", "suspension",
                       "your account", "verify your account", "agree", 'bank', 'dear', "update", "confirm",
                       "customer", "client", "suspend", "restrict", "verify", "login", "ssn", 'username', 'click',
                       'log', 'inconvenient', 'alert', 'paypal']

_DEFAULT_BLACKLIST = {key: 0 for key in BODY_WORD_BLACKLIST}


@register_feature(FeatureType.EMAIL_BODY, 'blacklisted_words_body', default_value=_DEFAULT_BLACKLIST)
def blacklisted_words_body(body: EmailBody):

    blacklist_body_count = {}

    if body.text is None:
        return _DEFAULT_BLACKLIST

    for word in BODY_WORD_BLACKLIST:
        word_count = len(re.findall(word, body.text, re.IGNORECASE))
        blacklist_body_count[word] = word_count
    return blacklist_body_count


@register_feature(FeatureType.EMAIL_BODY, 'number_of_scripts')
def number_scripts(body: EmailBody):
    if not body.is_html:
        return 0
    soup = BeautifulSoup(body.text, "html.parser")
    return len(soup.find_all('script'))


@register_feature(FeatureType.EMAIL_BODY, 'number_of_img_links')
def number_img_links(body: EmailBody):
    if not body.is_html:
        return 0
    soup = BeautifulSoup(body.html, "html.parser")
    return len(soup.find_all('img'))


# Source: http://www.viviancook.uk/Words/StructureWordsList.htm
_FUNCTION_WORDS = {"a", "about", "above", "after", "again", "against", "ago", "ahead", "all", "almost",
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


@register_feature(FeatureType.EMAIL_BODY, 'function_words_count')
def function_words_counts(body: EmailBody):
    if body.text is None:
        return 0
    function_words_count = 0
    for word in body.text.split(' '):
        if word in _FUNCTION_WORDS:
            function_words_count = +1
    return function_words_count


# region texststat metrics
@register_feature(FeatureType.EMAIL_BODY, 'flesh_read_score')
def flesh_read_score(body: EmailBody):
    return textstat.flesch_reading_ease(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'smog_index')
def smog_index(body: EmailBody):
    return textstat.smog_index(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'flesh_kincaid_score')
def flesh_kincaid_score(body: EmailBody):
    return textstat.flesch_kincaid_grade(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'coleman_liau_index')
def coleman_liau_index(body: EmailBody):
    return textstat.coleman_liau_index(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'automated_readability_index')
def automated_readability_index(body: EmailBody):
    return textstat.automated_readability_index(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'dale_chall_readability_score')
def dale_chall_readability_score(body: EmailBody):
    return textstat.dale_chall_readability_score(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'difficult_words')
def difficult_words(body: EmailBody):
    return textstat.difficult_words(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'linsear_score')
def linsear_score(body: EmailBody):
    return textstat.linsear_write_formula(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'gunning_fog')
def gunning_fog(body: EmailBody):
    return textstat.gunning_fog(body.text)


@register_feature(FeatureType.EMAIL_BODY, 'text_standard')
def text_standard(body: EmailBody):
    return textstat.text_standard(body.text, float_output=True)

# endregion
