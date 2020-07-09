import re
import sys

from . import helpers
from ...reflection.core import register_feature, FeatureType
from phishbench.input.email_input.models import EmailHeader


@register_feature(FeatureType.HEADER,'mime_version')
def email_header_mime_version(header: EmailHeader):
    return header.mime_version


@register_feature(FeatureType.HEADER, 'header_file_size')
def email_header_size_in_bytes(header: EmailHeader):
    return sys.getsizeof(header.header.encode("utf-8"))


@register_feature(FeatureType.HEADER, 'return_path')
def email_header_return_path(header: EmailHeader):
    return header.return_path


@register_feature(FeatureType.HEADER, 'X-mailer')
def email_header_x_mailer(header: EmailHeader):
    return header.x_mailer


@register_feature(FeatureType.HEADER, 'X_originating_hostname')
def email_header_x_originating_hostname(header: EmailHeader):
    return header.x_originating_hostname


@register_feature(FeatureType.HEADER, 'X_originating_hostname')
def email_header_x_originating_ip(header: EmailHeader):
    return header.x_originating_ip


@register_feature(FeatureType.HEADER, 'X_virus_scanned')
def email_header_x_virus_scanned(header: EmailHeader):
    return header.x_virus_scanned


@register_feature(FeatureType.HEADER, 'X_Spam_flag')
def x_spam_flag(header: EmailHeader):
    return header.x_spam_flag

@register_feature(FeatureType.HEADER, 'received_count')
def email_header_received_count(header: EmailHeader):
    if header.received is None:
        return 0
    return len(header.received)


@register_feature(FeatureType.HEADER, 'Authentication_Results_SPF_Pass')
def email_header_authentication_results_spf_pass(header: EmailHeader):
    if "spf=pass" in header.authentication_results:
        return True
    else:
        return False


@register_feature(FeatureType.HEADER, 'Authentication_Results_DKIM_Pass')
def email_header_authentication_results_dkim_pass(header: EmailHeader):
    if "dkim=pass" in header.authentication_results:
        return True
    else:
        return False


@register_feature(FeatureType.HEADER, 'X_Origininal_Authentication_results')
def email_header_x_original_authentication_results(header: EmailHeader):
    return header.x_original_authentication_results


@register_feature(FeatureType.HEADER, 'Received_SPF')
def email_header_received_spf(header: EmailHeader):
    return header.received_spf


@register_feature(FeatureType.HEADER, 'Dkim_Signature_Exists')
def email_header_dkim_signature_exists(header: EmailHeader):
    return header.dkim_signed


@register_feature(FeatureType.HEADER, 'compare_sender_domain_message_id_domain')
def email_header_compare_sender_domain_message_id_domain(header: EmailHeader):
    if header.message_id is not None:
        message_id_domain = header.message_id.split("@")[1]
    else:
        message_id_domain = None
    if header.sender_email_address is not None:
        sender_domain = header.sender_email_address.split("@")[1]
    else:
        sender_domain = None
    return sender_domain == message_id_domain


@register_feature(FeatureType.HEADER, 'compare_sender_return')
def email_header_compare_sender_return(header: EmailHeader):
    return header.sender_email_address == header.return_path


@register_feature(FeatureType.HEADER, 'blacklisted_words_subject')
def email_header_blacklisted_words_subject(header: EmailHeader):
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


@register_feature(FeatureType.HEADER, 'Number_Cc')
def email_header_number_cc(header: EmailHeader):
    return len(header.cc)


@register_feature(FeatureType.HEADER, 'Number_BCC')
def email_header_number_bcc(header: EmailHeader):
    return len(header.bcc)


@register_feature(FeatureType.HEADER, 'Number_to')
def email_header_number_to(header: EmailHeader):
    return len(header.to)


# region Subject features

@register_feature(FeatureType.HEADER, "number_of_words_subject")
def email_header_number_of_words_subject(header: EmailHeader):
    if header.subject:
        return len(re.findall(r'\w+', header.subject))
    return 0


@register_feature(FeatureType.HEADER, "number_of_characters_subject")
def email_header_number_of_characters_subject(header: EmailHeader):
    if header.subject:
        return len(re.findall(r'\w', header.subject))
    return 0


@register_feature(FeatureType.HEADER, "number_of_special_characters_subject")
def email_header_number_of_special_characters_subject(header: EmailHeader):
    subject = header.subject

    if subject is None:
        return 0

    num_char = len(subject)
    num_space = len(re.findall(r' ', subject))
    return len(subject) - num_char - num_space


@register_feature(FeatureType.HEADER, "is_forward")
def email_header_fwd_in_subject(header: EmailHeader):
    fw_regex = re.compile(r"^fw:", flags=re.IGNORECASE)
    if header.subject:
        return fw_regex.match(header.subject) is not None
    return False


@register_feature(FeatureType.HEADER, "is_reply")
def email_header_is_reply(header: EmailHeader):
    re_regex = re.compile(r"^re:", flags=re.IGNORECASE)
    if header.subject:
        return re_regex.match(header.subject) is not None
    return False


@register_feature(FeatureType.HEADER, "vocab_richness_subject")
def email_header_vocab_richness_subject(header: EmailHeader):
    if header.subject:
        return helpers.yule(header.subject)
    return 0

# endregion
