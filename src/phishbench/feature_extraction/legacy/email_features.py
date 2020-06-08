import copy
import email as em
import os
import pickle
import re
import sys
import time

from bs4 import BeautifulSoup
from tqdm import tqdm

from ... import Features
from ... import Features_Support
from ...utils import Globals
from ...input import input


def extract_dataset_features(legit_datset_folder, phish_dataset_folder):
    Globals.logger.info("Extracting email features. Legit: %s Phish: %s", legit_datset_folder,phish_dataset_folder)

    feature_list_dict_train = []
    extraction_time_dict_train = []
    num_legit, legit_corpus = extract_email_features(legit_datset_folder, feature_list_dict_train,
                                                     extraction_time_dict_train)
    num_phish, phish_corpus = extract_email_features(phish_dataset_folder, feature_list_dict_train,
                                                                extraction_time_dict_train)

    Features_Support.Cleaning(feature_list_dict_train)

    labels_train = [0] * num_legit + [1] * num_phish
    corpus_train = legit_corpus + phish_corpus
    return feature_list_dict_train, labels_train, corpus_train


def extract_email_features(dataset_path, feature_list_dict, time_list_dict):
    '''

    Parameters
    ----------
    dataset_path
    feature_list_dict
    time_list_dict

    Returns
    -------
    int:
        The number of additional emails added
    List[str]:
        The corpus of email
    '''
    print("Extracting Email features from {}".format(dataset_path))
    Globals.logger.info("Extracting Email features from {}".format(dataset_path))
    corpus_files = input.enumerate_folder_files(dataset_path)
    corpus = input.read_corpus(corpus_files, "ISO-8859-1")
    initial_size = len(feature_list_dict)
    for file_path, file_contents in tqdm(corpus.items()):
        Globals.logger.info(file_path)

        dict_features, dict_time = email_features(file_contents)
        feature_list_dict.append(dict_features)
        time_list_dict.append(dict_time)

        # dump_features_emails(file_path, dict_features, features_output,
        #                      feature_list_dict, dict_time, extraction_time_dict)

        Globals.summary.write("filepath: {}\n\n".format(file_path))
        Globals.summary.write("features extracted for this file:\n")
        for feature_name, feature_time in dict_time.items():
            Globals.summary.write("{} \n".format(feature_name))
            Globals.summary.write("extraction time: {} \n".format(feature_time))
        Globals.summary.write("\n#######\n")
    count_files = len(feature_list_dict) - initial_size
    return count_files, list(corpus.values())


def email_features(raw_email):
    try:
        body_text, body_html, text_Html, test_text, num_attachment, content_disposition_list, content_type_list, \
        Content_Transfer_Encoding_list, file_extension_list, charset_list, size_in_Bytes = \
            extract_body(raw_email)
        Globals.logger.debug("extract_body >>>> Done")

        url_All = get_url(body_html)

        Globals.logger.debug("extract urls from body >>>> Done")

        (subject, sender_full, recipient_full, recipient_name, recipient_full_address, recipient_domain, message_id
         , sender_name, sender_full_address, sender_domain, return_addr, x_virus_scanned, x_spam_flag,
         x_originating_ip, x_mailer
         , x_originating_hostname, dkim_signature, received_spf, x_original_authentication_results,
         authentication_results
         , received, Cc, Bcc, To, MIME_version) = extract_header_fields(raw_email)

        Globals.logger.debug("extract_header_fields >>>> Done")

        dict_features, dict_time = single_email_features(body_text, body_html, text_Html, test_text, num_attachment,
                                                         content_disposition_list, content_type_list,
                                                         Content_Transfer_Encoding_list, file_extension_list,
                                                         charset_list, size_in_Bytes, subject, sender_full,
                                                         recipient_full, str(recipient_name), recipient_full_address,
                                                         recipient_domain, message_id, sender_name, sender_full_address,
                                                         sender_domain, return_addr, x_virus_scanned, x_spam_flag,
                                                         x_originating_ip, x_mailer, x_originating_hostname,
                                                         dkim_signature, received_spf,
                                                         x_original_authentication_results, authentication_results,
                                                         received, Cc, Bcc, To, MIME_version)
        Globals.logger.debug("Email features >>>>>>>>>>> Done")

        return dict_features, dict_time

    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        return {}, {}


def get_url(body):
    url_regex = re.compile('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', flags=re.IGNORECASE | re.MULTILINE)
    url = re.findall(url_regex, body)
    return url


def extract_header_fields(email):
    # with open(filepath,'r') as f:
    # email=f.read()

    email_address_regex = re.compile(r"<.*@[a-zA-Z0-9.\-_]*", flags=re.MULTILINE | re.IGNORECASE)
    email_address_name_regex = re.compile(r'"?.*"? <?', flags=re.MULTILINE | re.IGNORECASE)
    email_address_domain_regex = re.compile(r"@.*", flags=re.MULTILINE | re.IGNORECASE)

    try:
        msg = em.message_from_string(email)
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))

    try:
        subject = msg['Subject']
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        subject = "None"

    try:
        if msg['Return-Path'] != None:
            # return_addr=msg['Return-Path'].strip('>').strip('<')
            return_addr = 1
        else:
            return_addr = 0
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        return_addr = "None"

    try:
        sender_full = msg['From']
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        sender_full = "None"

    try:
        if re.findall(email_address_name_regex, sender_full) != []:
            sender_name = re.findall(email_address_name_regex, sender_full)[0].strip('"').strip(' <').strip('"')
        else:
            sender_name = "None"
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        sender_name = "None"
    # print(sender_name)

    try:
        if re.findall(email_address_regex, sender_full) != []:
            sender_full_address = re.findall(email_address_regex, sender_full)[0]
        else:
            sender_full_address = "None"
    except  Exception as e:
        Globals.logger.warning("exception: " + str(e))
        sender_full_address = "None"

    try:
        if re.findall(email_address_domain_regex, sender_full) != []:
            sender_domain = re.findall(email_address_domain_regex, sender_full)[0].strip('@').strip('>')
        else:
            sender_domain = "None"
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        sender_domain = "None"

    try:
        recipient_full = msg['To']
        if recipient_full is None:
            if msg['Delivered-To']:
                recipient_full = msg['Delivered-To']
            elif msg['X-Envelope-To']:
                recipient_full = msg['X-Envelope-To']
            else:
                recipient_full = "None"
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        recipient_full = "None"

    try:
        if re.findall(email_address_name_regex, recipient_full) != []:
            # recipient_name=re.findall(email_address_name_regex,recipient_full)
            recipient_name = re.findall(email_address_name_regex, recipient_full)[0].strip('"').strip(' <').strip('"')
        # if recipient_name != []:
        #    recipient_name=recipient_name[0].split('"')
        else:
            recipient_name = "None"
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        recipient_name = "None"

    try:
        recipient_full_address = re.findall(email_address_regex, recipient_full)
        for address in recipient_full_address:
            recipient_full_address[recipient_full_address.index(address)] = address.strip("<")
        # recipient_full_address[]=address.strip("<") for address in recipient_full_address
        if recipient_full_address != []:
            # recipient_full_address=recipient_full_address[0]
            # print(re.findall(email_address_domain,recipient_full_address)[0])
            recipient_domain = []
            for address in recipient_full_address:
                recipient_domain.append(re.findall(email_address_domain_regex, address)[0].strip("@"))
            # print("recipient_domain >>>>>>>{}".format(recipient_domain))
        else:
            recipient_full_address = "None"
            recipient_domain = "None"
            # if "undisclosed-recipients" in recipient_full:
            #   recipient_name='undisclosed-recipients'
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        recipient_full_address = "None"
        recipient_domain = "None"

        # recipient_name="undisclosed-recipients"

    # if 'undisclosed-recipients' in recipient_full:
    #     recipient_name='undisclosed-recipients'
    #    recipient_full_address = 'None'
    #   recipient_domain = "None"
    # elif ',' in recipient_full:
    #   recipient_full_address = recipient_full.split(',')[1]
    #  recipient_domain = recipient_full_address.split("@")[1]
    # recipient_name = "None"
    # else:
    #    recipient_name=recipient_full.split("<")[0].strip('"')
    #   recipient_full_address=recipient_full.split("<")[1].strip('>')
    #  recipient_domain=sender_full_address.split("@")[1]

    # print(str(recipient_name),recipient_full_address,recipient_domain)
    try:
        if msg['Message-Id'] != None:
            message_id = msg['Message-Id'].strip('>').strip('<')
        else:
            message_id = "None"
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        message_id = "None"

    try:
        if msg['X-mailer'] != None:
            # x_mailer=msg['X-mailer']
            x_mailer = 1
        else:
            x_mailer = 0
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        x_mailer = "None"

    try:
        if msg["X-originating-hostname"] != None:
            # x_originating_hostname = msg["X-originating-hostname"]
            x_originating_hostname = 1
        else:
            x_originating_hostname = 0
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        x_originating_hostname = "None"

    try:
        if msg["X-originating-ip"] != None:
            x_originating_ip = 1
        else:
            x_originating_ip = 0
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        x_originating_ip = "None"

    try:
        if msg["X-Spam_flag"] != None:
            x_spam_flag = 1
        else:
            x_spam_flag = 0
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        x_spam_flag = "None"

    try:
        if msg["X-virus-scanned"] != None:
            x_virus_scanned = 1
        else:
            x_virus_scanned = 0
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        x_virus_scanned = "None"

    try:
        if msg["DKIM-Signature"] != None:
            # dkim_signature=msg["DKIM-Signature"]
            dkim_signature = 1
        else:
            dkim_signature = 0
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        dkim_signature = 0

    try:
        if msg["Received-SPF"] != None:
            # received_spf=msg["Received-SPF"]
            received_spf = 1
        else:
            # received_spf="None"
            received_spf = 0
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        received_spf = 0

    try:
        if msg["X-Original-Authentication-Results"] != None:
            # x_original_authentication_results = msg["X-Original-Authentication-Results"]
            x_original_authentication_results = 1
        else:
            x_original_authentication_results = 0
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        x_original_authentication_results = "None"

    try:
        if msg["Authentication-Results"] != None:
            authentication_results = msg["Authentication-Results"]
        else:
            authentication_results = "None"
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        authentication_results = "None"

    try:
        if msg["Received"]:
            received = msg.get_all("Received")
            # print("received: {}".format(received))
        else:
            received = "None"
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        received = "None"

    try:
        if msg["Cc"]:
            Cc = msg["Cc"]
        else:
            Cc = "None"
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        Cc = "None"

    try:
        if msg["Bcc"]:
            Bcc = msg["Bcc"]
        else:
            Bcc = "None"
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        Bcc = "None"

    try:
        if msg["To"]:
            To = msg["To"]
        else:
            To = "None"
    except Exception as e:
        Globals.logger.warning("exception: " + str(e))
        To = "None"

    try:
        if msg['MIME-Version']:
            MIME_version = re.findall(r'\d.\d', msg['MIME-Version'])[0]
        else:
            MIME_version = 0
    except Exception as e:
        Globals.logger.exception(e)
        MIME_version = "None"

    # print(message_id)
    return subject, sender_full, recipient_full, recipient_name, recipient_full_address, recipient_domain, message_id, \
           sender_name, sender_full_address, sender_domain, return_addr, x_virus_scanned, x_spam_flag, x_originating_ip, x_mailer, \
           x_originating_hostname, dkim_signature, received_spf, x_original_authentication_results, authentication_results, \
           received, Cc, Bcc, To, MIME_version


def extract_header(email):
    msg = em.message_from_string(str(email))
    # print(msg.items())
    header = str(msg.items()).replace('{', '').replace('}', '').replace(': ', ':').replace(',', '')
    return header


def extract_body(email):
    # with open(filepath, 'r') as f:
    #   email=f.read()
    hex_regex = re.compile(r"0x[0-9]*,?", flags=re.IGNORECASE | re.MULTILINE)
    css_regex = re.compile(r'(<style type="text/css">.*</style>)|(<style>.*</style>)',
                           flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    msg = em.message_from_string(str(email))
    # The reason for encoding is that, in Python 3, some single-character strings will require multiple bytes to be represented. For instance: len('你'.encode('utf-8'))
    size_in_Bytes = sys.getsizeof(email.encode("utf-8"))
    # header=msg.items()
    body = ""
    # if msg.is_multipart():
    test_text = 0
    text_Html = 0
    body_text = ''
    body_html = ''
    content_type_list = []
    content_disposition_list = []
    num_attachment = 0
    charset_list = []
    Content_Transfer_Encoding_list = []
    file_extension_list = []
    for part in msg.walk():
        # print("in walk loop")
        ctype = part.get_content_type()
        # ctype = part.get('Content-Type')
        content_type_list.append(ctype)
        # print("ctype list {}".format(content_type_list))

        cdispo = str(part.get_content_disposition())
        cdispo_full = str(part.get('Content-Disposition'))
        filename = re.findall(r'(?!filename=)".*"', cdispo_full)
        if filename != []:
            file_extension = os.path.splitext(filename[0])[1]
            file_extension_list.append(file_extension)
        # print("file_extension_list {}".format(file_extension_list))
        # print("cdispo_full {}".format(test))
        content_disposition_list.append(cdispo)
        # print("Content-Disposition {}".format(content_disposition_list))
        if 'attachment' in cdispo:
            num_attachment = +1
        if part.get_content_charset():
            charset_list.append(part.get_content_charset())
        # print("Charsets list: {}".format(charset_list))
        # print(filepath + '_ ATTACHMENT :' +str(test_attachment))
        # skip any text/plain (txt) attachments
        ctransfer = str(part.get('Content-Transfer-Encoding'))
        Content_Transfer_Encoding_list.append(ctransfer)
        # print("Content-Transfer-Encoding list: {}".format(Content_Transfer_Encoding_list))
        if ctype == 'text/plain':
            # print("text/plain loop")
            # print("Charset: {}".format(part.get_content_charset()))
            try:
                body_text = part.get_payload(decode=True).decode(part.get_content_charset())
            except Exception as e:
                Globals.logger.exception(e)
                body_text = part.get_payload(decode=False)  # decode
            # body_text = part.get_payload(decode=False)
            # print("\n\n\n")
            # print("body_text_______________")
            # print(body_text)
            # print("\n\n\n")
            test_text = 1
        if ctype == 'text/html':
            # print("text/html loop")
            # print("Charset: {}".format(part.get_content_charset()))
            try:
                html = part.get_payload(decode=True).decode(part.get_content_charset())
            except Exception as e:
                Globals.logger.warning('Exception: {}'.format(e))
                html = part.get_payload(decode=False)
            # html=part.get_payload(decode=False)
            html = css_regex.sub('', str(html))
            soup = BeautifulSoup(html, 'html.parser')
            if body_text == '':
                # print("\n\n\n")
                body_text = soup.text
                body_text = hex_regex.sub('', body_text)
                # print("body_text_______________")
                # print("\n\n\n")
                # body_text=body_text.decode()
            # print("\n\n\n")
            # print("body_HTML_______________")
            body_html = str(soup)
            # print(body_html)
            # print("\n\n\n")
            # if hex_regex.search(body):
            #   hex_log.write(filepath+"\n")
            text_Html = 1
        else:
            body_text = part.get_payload(decode=True).decode("utf-8")

    return body_text, body_html, text_Html, test_text, num_attachment, content_disposition_list, content_type_list, \
           Content_Transfer_Encoding_list, file_extension_list, charset_list, size_in_Bytes


def dump_features_emails(header, list_features, features_output, list_dict, list_time, time_dict):
    Globals.logger.debug("list_features: " + str(len(list_features)))
    list_dict.append(copy.copy(list_features))
    time_dict.append(copy.copy(list_time))
    with open(features_output + "_feature_vector.pkl", 'ab') as feature_tracking:
        pickle.dump("email: " + str(header), feature_tracking)
        pickle.dump(list_features, feature_tracking)
    with open(features_output + "_feature_vector.txt", 'a+') as f:
        f.write("email: " + str(header) + '\n' + str(list_features).replace('{', '').replace('}', '').replace(': ',
                                                                                                              ':').replace(
            ',', '') + '\n\n')
    with open(features_output + "_time_stats.txt", 'a+') as f:
        f.write("email: " + str(header) + '\n' + str(list_time).replace('{', '').replace('}', '').replace(': ',
                                                                                                          ':').replace(
            ',', '') + '\n\n')


def single_email_features(body_text, body_html, text_Html, test_text, num_attachment, content_disposition_list,
                          content_type_list
                          , Content_Transfer_Encoding_list, file_extension_list, charset_list, size_in_Bytes, subject,
                          sender_full, recipient_full, recipient_name, recipient_full_address, recipient_domain,
                          message_id
                          , sender_name, sender_full_address, sender_domain, return_addr, x_virus_scanned, x_spam_flag,
                          x_originating_ip, x_mailer
                          , x_originating_hostname, dkim_signature, received_spf, x_original_authentication_results,
                          authentication_results
                          , received, Cc, Bcc, To, MIME_version):
    dict_features = {}
    dict_time = {}
    if Globals.config["Email_Features"]["extract header features"] == "True":
        Globals.logger.debug("Extracting Header Features")
        Features.Email_Header_return_path(return_addr, dict_features, dict_time)
        Features.Email_Header_X_mailer(x_mailer, dict_features, dict_time)
        Features.Email_Header_X_originating_hostname(x_originating_hostname, dict_features, dict_time)
        Features.Email_Header_X_originating_ip(x_originating_ip, dict_features, dict_time)
        Features.Email_Header_X_spam_flag(x_spam_flag, dict_features, dict_time)
        Features.Email_Header_X_virus_scanned(x_virus_scanned, dict_features, dict_time)
        Features.Email_Header_X_Origininal_Authentication_results(x_original_authentication_results, dict_features,
                                                                  dict_time)
        Features.Email_Header_Received_SPF(received_spf, dict_features, dict_time)
        Features.Email_Header_Dkim_Signature_Exists(dkim_signature, dict_features, dict_time)
        Features.Email_Header_number_of_words_subject(subject, dict_features, dict_time)
        Features.Email_Header_number_of_characters_subject(subject, dict_features, dict_time)
        Features.Email_Header_number_of_special_characters_subject(subject, dict_features, dict_time)
        Features.Email_Header_binary_fwd(subject, dict_features, dict_time)
        Features.Email_Header_vocab_richness_subject(subject, dict_features, dict_time)
        Features.Email_Header_compare_sender_return(sender_full_address, return_addr, dict_features, dict_time)
        Features.Email_Header_compare_sender_domain_message_id_domain(sender_domain, message_id, dict_features,
                                                                      dict_time)
        # Features.Content_Disposition(cdispo, list_features, list_time)
        # Globals.logger.debug("Content_Disposition")
        Features.Email_Header_Number_Cc(Cc, dict_features, dict_time)
        Features.Email_Header_Number_Bcc(Bcc, dict_features, dict_time)
        Features.Email_Header_Number_To(To, dict_features, dict_time)
        Features.Email_Header_Num_Content_type(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Charset(charset_list, dict_features, dict_time)
        Features.Email_Header_Num_Unique_Charset(charset_list, dict_features, dict_time)
        Features.Email_Header_MIME_Version(MIME_version, dict_features, dict_time)
        Features.Email_Header_Num_Unique_Content_type(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Unique_Content_Disposition(content_disposition_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Disposition(content_disposition_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_text_plain(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_text_html(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Disposition(content_disposition_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_text_plain(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_text_html(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Multipart_Encrypted(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Multipart_Mixed(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Multipart_form_data(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Multipart_byterange(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Multipart_Parallel(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Multipart_Report(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Multipart_Alternative(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Multipart_Digest_Num(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Multipart_Signed_Num(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Multipart_X_Mixed_Replaced(content_type_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Charset_us_ascii(charset_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Charset_utf_8(charset_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Charset_utf_7(charset_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Charset_gb2312(charset_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Charset_shift_jis(charset_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Charset_koi(charset_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Type_Charset_iso2022_jp(charset_list, dict_features, dict_time)
        Features.Email_Header_Num_Attachment(num_attachment, dict_features, dict_time)
        Features.Email_Header_Num_Unique_Attachment_types(file_extension_list, dict_features, dict_time)
        Features.Email_Header_Num_Content_Transfer_Encoding(Content_Transfer_Encoding_list, dict_features, dict_time)
        Features.Email_Header_Num_Unique_Content_Transfer_Encoding(Content_Transfer_Encoding_list, dict_features,
                                                                   dict_time)
        Features.Email_Header_Num_Content_Transfer_Encoding_7bit(Content_Transfer_Encoding_list, dict_features,
                                                                 dict_time)
        Features.Email_Header_Num_Content_Transfer_Encoding_8bit(Content_Transfer_Encoding_list, dict_features,
                                                                 dict_time)
        Features.Email_Header_Num_Content_Transfer_Encoding_binary(Content_Transfer_Encoding_list, dict_features,
                                                                   dict_time)
        Features.Email_Header_Num_Content_Transfer_Encoding_quoted_printable(Content_Transfer_Encoding_list,
                                                                             dict_features, dict_time)
        Features.Email_Header_Num_Unique_Attachment_types(file_extension_list, dict_features, dict_time)
        Features.Email_Header_size_in_Bytes(size_in_Bytes, dict_features, dict_time)
        Features.Email_Header_Received_count(received, dict_features, dict_time)
        Features.Email_Header_Authentication_Results_SPF_Pass(authentication_results, dict_features, dict_time)
        Features.Email_Header_Authentication_Results_DKIM_Pass(authentication_results, dict_features, dict_time)
        Features.Email_Header_Test_Html(text_Html, dict_features, dict_time)
        Features.Email_Header_Test_Text(test_text, dict_features, dict_time)
        Features.Email_Header_blacklisted_words_subject(subject, dict_features, dict_time)

    if Globals.config["Email_Features"]["extract body features"] == "True":
        Globals.logger.debug("Extracting Body features")
        Features.Email_Body_flesh_read_score(body_text, dict_features, dict_time)
        Features.Email_Body_smog_index(body_text, dict_features, dict_time)
        Features.Email_Body_flesh_kincaid_score(body_text, dict_features, dict_time)
        Features.Email_Body_coleman_liau_index(body_text, dict_features, dict_time)
        Features.Email_Body_automated_readability_index(body_text, dict_features, dict_time)
        Features.Email_Body_dale_chall_readability_score(body_text, dict_features, dict_time)
        Features.Email_Body_difficult_words(body_text, dict_features, dict_time)
        Features.Email_Body_linsear_score(body_text, dict_features, dict_time)
        Features.Email_Body_gunning_fog(body_text, dict_features, dict_time)
        # Features.html_in_body(body, list_features, list_time)
        # print("html_in_body")
        Features.Email_Body_number_of_words_body(body_text, dict_features, dict_time)
        Features.Email_Body_number_of_characters_body(body_text, dict_features, dict_time)
        Features.Email_Body_number_of_special_characters_body(body_text, dict_features, dict_time)
        Features.Email_Body_vocab_richness_body(body_text, dict_features, dict_time)
        Features.Email_Body_number_of_html_tags_body(body_html, dict_features, dict_time)
        Features.Email_Body_number_of_unique_words_body(body_text, dict_features, dict_time)
        Features.Email_Body_number_unique_chars_body(body_text, dict_features, dict_time)
        Features.Email_Body_end_tag_count(body_html, dict_features, dict_time)
        Features.Email_Body_open_tag_count(body_html, dict_features, dict_time)
        Features.Email_Body_recipient_name_body(body_text, recipient_name, dict_features, dict_time)
        Features.Email_Body_on_mouse_over(body_html, dict_features, dict_time)
        Features.Email_Body_count_href_tag(body_html, dict_features, dict_time)
        Features.Email_Body_Function_Words_Count(body_text, dict_features, dict_time)
        Features.Email_Body_Number_Of_Img_Links(body_html, dict_features, dict_time)
        Features.Email_Body_blacklisted_words_body(body_text, dict_features, dict_time)
        Features.Email_Body_Number_Of_Scripts(body_html, dict_features, dict_time)

    return dict_features, dict_time


def email_url_features(url_All, sender_domain, list_features, list_time):
    if Globals.config["Email_Features"]["extract body features"] == "True":
        Globals.logger.debug("Extracting email URL features")
        Features.Email_URL_Number_Url(url_All, list_features, list_time)
        Features.Email_URL_Number_Diff_Domain(url_All, list_features, list_time)
        Features.Email_URL_Number_link_at(url_All, list_features, list_time)
        Features.Email_URL_Number_link_sec_port(url_All, list_features, list_time)
    #
    # Features.Number_link_IP(url_All, list_features, list_time)
    # Globals.logger.debug(Number_link_IP)
    # Features.Number_link_HTTPS(url_All, list_features, list_time)
    # Globals.logger.debug(Number_link_HTTPS)
    # Features.Number_Domain_Diff_Sender(url_All, sender_domain, list_features, list_time)
    # print(Number_Domain_Diff_Sender)
    # Features.Number_Link_Text(url_All, list_features, list_time)
    # print(Number_Link_Text)
    # Features.Number_link_port_diff_8080(url_All, list_features, list_time)
    # print(Number_link_port_diff_8080)
