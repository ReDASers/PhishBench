"""
Internal header features
"""
import re
import sys

from . import helpers
from ...reflection import register_feature, FeatureType
from ....input.email_input.models import EmailHeader


@register_feature(FeatureType.EMAIL_HEADER, 'mime_version')
def mime_version(header: EmailHeader):
    """
    The MIME version
    """
    return header.mime_version


@register_feature(FeatureType.EMAIL_HEADER, 'header_file_size')
def size_in_bytes(header: EmailHeader):
    """
    The size of the header in bytes
    """
    return sys.getsizeof(header.header.encode("utf-8"))


@register_feature(FeatureType.EMAIL_HEADER, 'return_path')
def return_path(header: EmailHeader):
    """
    The return path of the email
    """
    return header.return_path


@register_feature(FeatureType.EMAIL_HEADER, 'X-mailer')
def x_mailer(header: EmailHeader):
    """
    The x-mailer of the email
    """
    return header.x_mailer


@register_feature(FeatureType.EMAIL_HEADER, 'x_originating_hostname')
def x_originating_hostname(header: EmailHeader):
    """
    The x-originating-hostname header of the email
    """
    return header.x_originating_hostname


@register_feature(FeatureType.EMAIL_HEADER, 'x_originating_ip')
def x_originating_ip(header: EmailHeader):
    """
    The x-originating-ip header of the email
    """
    return header.x_originating_ip


@register_feature(FeatureType.EMAIL_HEADER, 'x_virus_scanned')
def x_virus_scanned(header: EmailHeader):
    """
    Whether or not the x-virus-scanned header is in the email
    """
    return header.x_virus_scanned


@register_feature(FeatureType.EMAIL_HEADER, 'x_spam_flag')
def x_spam_flag(header: EmailHeader):
    """
    Whether or not the x-spam-flag header is in the email
    """
    return header.x_spam_flag


@register_feature(FeatureType.EMAIL_HEADER, 'received_count')
def received_count(header: EmailHeader):
    """
    The number of Received headers
    """
    return len(header.received)


@register_feature(FeatureType.EMAIL_HEADER, 'authentication_results_spf_pass')
def authentication_results_spf_pass(header: EmailHeader):
    """
    Whether or not `spf=pass` is in the authentication results
    """
    return "spf=pass" in header.authentication_results


@register_feature(FeatureType.EMAIL_HEADER, 'authentication_results_dkim_pass')
def authentication_results_dkim_pass(header: EmailHeader):
    """
    Whether or not `dkim=` is in the authentication results
    """
    return "dkim=pass" in header.authentication_results


@register_feature(FeatureType.EMAIL_HEADER, 'has_x_original_authentication_results')
def x_original_authentication_results(header: EmailHeader):
    """
    Whether or not the email has the X-Original-Authentication-Results header
    """
    return header.x_original_authentication_results


@register_feature(FeatureType.EMAIL_HEADER, 'has_received_spf')
def has_received_spf(header: EmailHeader):
    """
    Whether or not the email has the Recieved-SPF header
    """
    return header.received_spf


@register_feature(FeatureType.EMAIL_HEADER, 'has_dkim_signature')
def has_dkim_signature(header: EmailHeader):
    """
    Whether or not the email has the DKIM-Signature header
    """
    return header.dkim_signed


@register_feature(FeatureType.EMAIL_HEADER, 'compare_sender_domain_message_id_domain')
def compare_sender_domain_message_id_domain(header: EmailHeader):
    """
    Whether or not the domain for the sender address and the message id is the same

    """
    if header.message_id is not None and '@' in header.message_id:
        message_id_domain = header.message_id.split("@")[1]
    else:
        message_id_domain = None
    if header.sender_email_address is not None and '@' in header.message_id:
        sender_domain = header.sender_email_address.split("@")[1]
    else:
        sender_domain = None
    return sender_domain == message_id_domain


@register_feature(FeatureType.EMAIL_HEADER, 'compare_sender_return')
def compare_sender_return(header: EmailHeader):
    """
    Whether or not the return path and the sender address are the same
    """
    return header.sender_email_address == header.return_path


@register_feature(FeatureType.EMAIL_HEADER, 'blacklisted_words_subject')
def blacklisted_words_subject(header: EmailHeader):
    """
    Number of times the blacklisted words in `["urgent", "account", "closing", "act now", "click here", "limitied",
    "suspension", "your account", "verify your account", "agree", 'bank', 'dear', "update", "confirm", "customer",
    "client", "Suspend", "restrict", "verify", "login", "ssn", 'username','click', 'log', 'inconvenien', 'alert',
    'paypal']` appear in the subject.
    """
    blacklist_subject = ["urgent", "account", "closing", "act now", "click here", "limitied", "suspension",
                         "your account", "verify your account", "agree", 'bank', 'dear', "update", "confirm",
                         "customer", "client", "Suspend", "restrict", "verify", "login", "ssn", 'username',
                         'click', 'log', 'inconvenien', 'alert', 'paypal']
    result_dict = {}
    if header.subject is None:
        for word in blacklist_subject:
            result_dict[word] = 0
        return result_dict

    for word in blacklist_subject:
        word_count = len(re.findall(word, header.subject, re.IGNORECASE))
        result_dict[word] = word_count
    return result_dict


@register_feature(FeatureType.EMAIL_HEADER, 'number_cc')
def number_cc(header: EmailHeader):
    """
    Number of addresses in the CC field
    """
    return len(header.cc)


@register_feature(FeatureType.EMAIL_HEADER, 'number_bcc')
def number_bcc(header: EmailHeader):
    """
    Number of addresses in the BCC field
    """
    return len(header.bcc)


@register_feature(FeatureType.EMAIL_HEADER, 'number_to')
def number_to(header: EmailHeader):
    """
    Number of addresses in the to field
    """
    return len(header.to)


# region Subject features

@register_feature(FeatureType.EMAIL_HEADER, "number_of_words_subject")
def number_of_words_subject(header: EmailHeader):
    """
    The number of words in the subject
    """
    if header.subject:
        return len(re.findall(r'\w+', header.subject))
    return 0


@register_feature(FeatureType.EMAIL_HEADER, "number_of_characters_subject")
def number_of_characters_subject(header: EmailHeader):
    """
    The number of characters in the subject
    """
    if header.subject:
        return len(re.findall(r'\w', header.subject))
    return 0


@register_feature(FeatureType.EMAIL_HEADER, "number_of_special_characters_subject")
def number_of_special_characters_subject(header: EmailHeader):
    """
    The number of special characters in the subject
    """
    if header.subject:
        return len(re.findall(r'_|[^\w\s]', header.subject))
    return 0


@register_feature(FeatureType.EMAIL_HEADER, "is_forward")
def fwd_in_subject(header: EmailHeader):
    """
    Whether or not "fw:" is in the subject
    """
    fw_regex = re.compile(r"^fw:", flags=re.IGNORECASE)
    if header.subject:
        return fw_regex.match(header.subject) is not None
    return False


@register_feature(FeatureType.EMAIL_HEADER, "is_reply")
def is_reply(header: EmailHeader):
    """
    Whether or not "re:" is in the subject
    """
    re_regex = re.compile(r"^re:", flags=re.IGNORECASE)
    if header.subject:
        return re_regex.match(header.subject) is not None
    return False


@register_feature(FeatureType.EMAIL_HEADER, "vocab_richness_subject")
def vocab_richness_subject(header: EmailHeader):
    """
    The vocabulary richness (yule) of the subject
    """
    if header.subject:
        return helpers.yule(header.subject)
    return 0

# endregion
