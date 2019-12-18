import numpy as np
import re
import sys
import os, os.path
from itertools import groupby
import nltk
from textstat.textstat import textstat
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import email as em
import string
import math
from textblob import TextBlob as tb
from scipy.sparse.csr import csr_matrix
from urllib.parse import urlparse
from scipy import stats
from scipy.sparse import csc_matrix
from scipy.sparse import isspmatrix_csc
from scipy.sparse import isspmatrix_csr
from scipy.spatial import distance
import tldextract
import time
import pandas as pd
from bs4 import BeautifulSoup
import pickle
import json
#import User_options
import Download_url
import Features
import Tfidf
import timeit
from sklearn import preprocessing
import configparser
#from collections import deque
import Features
from sklearn.feature_extraction import DictVectorizer
import logging
import traceback
import ntpath
import copy
logger = logging.getLogger('root')

config=configparser.ConfigParser()
config.read('Config_file.ini')
#import user_options

import base64

#from collections import deque

#from bs4 import BeautifulSoup


PAYLOAD_NOT_FOUND = False # for filtering

## list of function words: http://www.viviancook.uk/Words/StructureWordsList.htm
Function_words_list=["a", "about", "above", "after", "again", "against", "ago", "ahead", "all", "almost", "almost", "along", "already", "also", "", "although", "always", "am", "among", "an", "and", "any", "are", "aren't", "around", "as", "at", "away", "backward", "backwards", "be", "because", "before", "behind", "below", "beneath", "beside", "between", "both", "but", "by", "can", "cannot", "can't", "cause", "'cos", "could", "couldn't", "'d", "had", "despite", "did", "didn't", "do", "does", "doesn't", "don't", "down", "during", "each", "either", "even", "ever", "every", "except", "for", "faw", "forward", "from", "frm", "had", "hadn't", "has", "hasn't", "have", "hv", "haven't", "he", "hi", "her", "here", "hers", "herself", "him", "hm", "himself", "his", "how", "however", "I", "if", "in", "inside", "inspite", "instead", "into", "is", "isn't", "it", "its", "itself", "just", "'ll", "will", "shall", "least", "less", "like", "'m", "them", "many", "may", "mayn't", "me", "might", "mightn't", "mine", "more", "most", "much", "must", "mustn't", "my", "myself", "near", "need", "needn't", "needs", "neither", "never", "no", "none", "nor", "not", "now", "of", "off", "often", "on", "once", "only", "onto", "or", "ought", "oughtn't", "our", "ours", "ourselves", "out", "outside", "over", "past", "perhaps", "quite", "'re", "rather", "'s", "", "seldom", "several", "shall", "shan't", "she", "should", "shouldn't", "since", "so", "some", "sometimes", "soon", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "therefore", "these", "they", "this", "those", "though", "", "through", "thus", "till", "to", "together", "too", "towards", "under", "unless", "until", "up", "upon", "us", "used", "usedn't", "usen't", "usually", "'ve", "very", "was", "wasn't", "we", "well", "were", "weren't", "what", "when", "where", "whether", "which", "while", "who", "whom", "whose", "why", "will", "with", "without", "won't", "would", "wouldn't", "yet", "you", "your", "yours", "yourself", "yourselves"]
''' Returns frequency of function words. Source:  https://stackoverflow.com/questions/5819840/calculate-frequency-of-function-words'''
def get_func_word_freq(words,funct_words):
    fdist = nltk.FreqDist([funct_word for funct_word in funct_words if funct_word in words])
    funct_freq = {}
    for key,value in fdist.iteritems():
        funct_freq[key] = value
    return funct_freq

''' Read LIWC 2007 English dictionary and extract function words '''
def load_liwc_funct():
    funct_words = set()
    data_file = open(liwc_dict_file, 'rb')
    lines = data_file.readlines()
    for line in lines:
        row = line.rstrip().split("\t")
        if '1' in row:
            if row[0][-1:] == '*' :
                funct_words.add(row[0][:-1])
            else :
                funct_words.add(row[0])
    return list(funct_words)

################# Vocabulary richness https://swizec.com/blog/measuring-vocabulary-richness-with-python/swizec/2528
def words(entry):
    return filter(lambda w: len(w) > 0,
                  [w.strip("0123456789!:,.?(){}[]") for w in entry.split()])

def yule(entry):
    # yule's I measure (the inverse of yule's K measure)
    # higher number is higher diversity - richer vocabulary
    d = {}
    stemmer = PorterStemmer()
    for w in words(entry):
        w = stemmer.stem(w).lower()
        try:
            d[w] += 1
        except KeyError:
            d[w] = 1

    M1 = float(len(d))
    M2 = sum([len(list(g))*(freq**2) for freq,g in groupby(sorted(d.values()))])
    try:
        return (M1*M1)/(M2-M1)
    except ZeroDivisionError:
        return 0
################


def extract_header_fields(email):
    #with open(filepath,'r') as f:
        #email=f.read()

    email_address_regex=re.compile(r"<.*@[a-zA-Z0-9.\-_]*", flags=re.MULTILINE|re.IGNORECASE)
    email_address_name_regex=re.compile(r'"?.*"? <?', flags=re.MULTILINE|re.IGNORECASE)
    email_address_domain_regex=re.compile(r"@.*", flags=re.MULTILINE|re.IGNORECASE)

    try:
        msg = em.message_from_string(email)
    except Exception as e:
        logger.warning("exception: " + str(e))

    try:
        subject=msg['Subject']
    except Exception as e:
        logger.warning("exception: " + str(e))
        subject="None"

    try:
        if msg['Return-Path'] != None:
            #return_addr=msg['Return-Path'].strip('>').strip('<')
            return_addr=1
        else:
            return_addr=0
    except Exception as e:
        logger.warning("exception: " + str(e))
        return_addr="None"

    try:
        sender_full=msg['From']
    except Exception as e:
        logger.warning("exception: " + str(e))
        sender_full="None"

    try:
        if re.findall(email_address_name_regex,sender_full)!=[]:
            sender_name=re.findall(email_address_name_regex,sender_full)[0].strip('"').strip(' <').strip('"')
        else:
            sender_name="None"
    except Exception as e:
        logger.warning("exception: " + str(e))
        sender_name="None"
    #print(sender_name)

    try:
        if re.findall(email_address_regex,sender_full)!=[]:
            sender_full_address=re.findall(email_address_regex,sender_full)[0]
        else:
            sender_full_address="None"
    except  Exception as e:
        logger.warning("exception: " + str(e))
        sender_full_address="None"

    try:
        if re.findall(email_address_domain_regex,sender_full)!=[]:
            sender_domain= re.findall(email_address_domain_regex,sender_full)[0].strip('@').strip('>')
        else:
            sender_domain="None"
    except Exception as e:
        logger.warning("exception: " + str(e))
        sender_domain="None"

    try:
        recipient_full=msg['To']
        if recipient_full==None:
            if msg['Delivered-To']:
                recipient_full=msg['Delivered-To']
            elif msg['X-Envelope-To']:
                recipient_full = msg['X-Envelope-To']
            else:
                recipient_full="None"
    except Exception as e:
        logger.warning("exception: " + str(e))
        recipient_full="None"

    try:
        if re.findall(email_address_name_regex,recipient_full) != []:
        #recipient_name=re.findall(email_address_name_regex,recipient_full)
            recipient_name=re.findall(email_address_name_regex,recipient_full)[0].strip('"').strip(' <').strip('"')
        #if recipient_name != []:
        #    recipient_name=recipient_name[0].split('"')
        else:
            recipient_name="None"
    except Exception as e:
        logger.warning("exception: " + str(e))
        recipient_name="None"

    try:
        recipient_full_address=re.findall(email_address_regex,recipient_full)
        for address in recipient_full_address:
            recipient_full_address[recipient_full_address.index(address)]=address.strip("<")
        #recipient_full_address[]=address.strip("<") for address in recipient_full_address
        if recipient_full_address!=[]:
            #recipient_full_address=recipient_full_address[0]
            #print(re.findall(email_address_domain,recipient_full_address)[0])
            recipient_domain=[]
            for address in recipient_full_address:
                recipient_domain.append(re.findall(email_address_domain_regex,address)[0].strip("@"))
            #print("recipient_domain >>>>>>>{}".format(recipient_domain))
        else:
            recipient_full_address = "None"
            recipient_domain = "None"
            #if "undisclosed-recipients" in recipient_full:
             #   recipient_name='undisclosed-recipients'
    except Exception as e:
        logger.warning("exception: " + str(e))
        recipient_full_address="None"
        recipient_domain="None"

        #recipient_name="undisclosed-recipients"

  # if 'undisclosed-recipients' in recipient_full:
   #     recipient_name='undisclosed-recipients'
    #    recipient_full_address = 'None'
     #   recipient_domain = "None"
    #elif ',' in recipient_full:
     #   recipient_full_address = recipient_full.split(',')[1]
      #  recipient_domain = recipient_full_address.split("@")[1]
       # recipient_name = "None"
   # else:
    #    recipient_name=recipient_full.split("<")[0].strip('"')
     #   recipient_full_address=recipient_full.split("<")[1].strip('>')
      #  recipient_domain=sender_full_address.split("@")[1]

    #print(str(recipient_name),recipient_full_address,recipient_domain)
    try:
        if msg['Message-Id']!=None:
            message_id=msg['Message-Id'].strip('>').strip('<')
        else:
            message_id="None"
    except Exception as e:
        logger.warning("exception: " + str(e))
        message_id="None"

    try:
        if msg['X-mailer'] != None:
            #x_mailer=msg['X-mailer']
            x_mailer=1
        else:
            x_mailer=0
    except Exception as e:
        logger.warning("exception: " + str(e))
        x_mailer="None"

    try:
        if msg["X-originating-hostname"] != None:
            #x_originating_hostname = msg["X-originating-hostname"]
            x_originating_hostname=1
        else:
            x_originating_hostname = 0
    except Exception as e:
        logger.warning("exception: " + str(e))
        x_originating_hostname="None"

    try:
        if msg["X-originating-ip"] != None:
            x_originating_ip= 1
        else:
            x_originating_ip= 0
    except Exception as e:
        logger.warning("exception: " + str(e))
        x_originating_ip="None"

    try:
        if msg["X-Spam_flag"] != None:
            x_spam_flag= 1
        else:
            x_spam_flag= 0
    except Exception as e:
        logger.warning("exception: " + str(e))
        x_spam_flag="None"

    try:
        if msg["X-virus-scanned"] != None:
            x_virus_scanned= 1
        else:
            x_virus_scanned= 0
    except Exception as e:
        logger.warning("exception: " + str(e))
        x_virus_scanned="None"

    try:
        if msg["DKIM-Signature"] != None:
            #dkim_signature=msg["DKIM-Signature"]
            dkim_signature=1
        else:
            dkim_signature=0
    except Exception as e:
        logger.warning("exception: " + str(e))
        dkim_signature = 0

    try:
        if msg["Received-SPF"] != None:
            #received_spf=msg["Received-SPF"]
            received_spf=1
        else:
            #received_spf="None"
            received_spf=0
    except Exception as e:
        logger.warning("exception: " + str(e))
        received_spf=0

    try:
        if msg["X-Original-Authentication-Results"] != None:
            #x_original_authentication_results = msg["X-Original-Authentication-Results"]
            x_original_authentication_results=1
        else:
            x_original_authentication_results =0
    except Exception as e:
        logger.warning("exception: " + str(e))
        x_original_authentication_results="None"

    try:
        if msg["Authentication-Results"] != None:
            authentication_results = msg["Authentication-Results"]
        else:
            authentication_results = "None"
    except Exception as e:
        logger.warning("exception: " + str(e))
        authentication_results="None"

    try:
        if msg["Received"] != []:
            received=msg.get_all("Received")
            #print("received: {}".format(received))
        else:
            received="None"
    except Exception as e:
        logger.warning("exception: " + str(e))
        received="None"

    try:
        if msg["Cc"]!=[]:
            Cc=msg["Cc"]
        else:
            Cc="None"
    except Exception as e:
        logger.warning("exception: "+ str(e))
        Cc="None"

    try:
        if msg["Bcc"]!=[]:
            Bcc=msg["Bcc"]
        else:
            Bcc="None"
    except Exception as e:
        logger.warning("exception: "+ str(e))
        Bcc="None"

    try:
        if msg["To"]!=[]:
            To=msg["To"]
        else:
            To="None"
    except Exception as e:
        logger.warning("exception: "+ str(e))
        To="None"

    try:
        if msg['MIME-Version'] != []:
            MIME_version=re.findall(r'\d.\d',msg['MIME-Version'])[0]
        else:
            MIME_version=0
    except Exception as e:
        logger.warning("exception: "+ str(e))
        MIME_version="None"

    #print(message_id)
    return subject, sender_full, recipient_full, recipient_name, recipient_full_address, recipient_domain,message_id,\
    sender_name,sender_full_address,sender_domain,return_addr,x_virus_scanned,x_spam_flag,x_originating_ip, x_mailer,\
     x_originating_hostname, dkim_signature, received_spf, x_original_authentication_results, authentication_results,\
      received, Cc, Bcc, To, MIME_version


############################
def extract_header(email):
    msg=em.message_from_string(str(email))
    #print(msg.items())
    header=str(msg.items()).replace('{','').replace('}','').replace(': ',':').replace(',','')
    return header

def extract_body(email):
    #with open(filepath, 'r') as f:
     #   email=f.read()
    hex_regex=re.compile(r"0x[0-9]*,?",flags=re.IGNORECASE|re.MULTILINE)
    css_regex=re.compile(r'(<style type="text/css">.*</style>)|(<style>.*</style>)',flags=re.IGNORECASE|re.MULTILINE|re.DOTALL)
    msg = em.message_from_string(str(email))
    #The reason for encoding is that, in Python 3, some single-character strings will require multiple bytes to be represented. For instance: len('你'.encode('utf-8'))
    size_in_Bytes=sys.getsizeof(email.encode("utf-8"))
    #header=msg.items()
    body = ""
    #if msg.is_multipart():
    test_text=0
    text_Html=0
    body_text=''
    body_html=''
    content_type_list=[]
    content_disposition_list=[]
    num_attachment=0
    charset_list=[]
    Content_Transfer_Encoding_list=[]
    file_extension_list=[]
    for part in msg.walk():
        #print("in walk loop")
        ctype = part.get_content_type()
        #ctype = part.get('Content-Type')
        content_type_list.append(ctype)
        #print("ctype list {}".format(content_type_list))

        cdispo = str(part.get_content_disposition())
        cdispo_full= str(part.get('Content-Disposition'))
        filename=re.findall(r'(?!filename=)".*"',cdispo_full)
        if filename != []:
            file_extension=os.path.splitext(filename[0])[1]
            file_extension_list.append(file_extension)
        #print("file_extension_list {}".format(file_extension_list))
        #print("cdispo_full {}".format(test))
        content_disposition_list.append(cdispo)
        #print("Content-Disposition {}".format(content_disposition_list))
        if 'attachment' in cdispo:
            num_attachment=+1
        if part.get_content_charset():
            charset_list.append(part.get_content_charset())
        #print("Charsets list: {}".format(charset_list))
        #print(filepath + '_ ATTACHMENT :' +str(test_attachment))
        # skip any text/plain (txt) attachments
        ctransfer=str(part.get('Content-Transfer-Encoding'))
        Content_Transfer_Encoding_list.append(ctransfer)
        #print("Content-Transfer-Encoding list: {}".format(Content_Transfer_Encoding_list))
        if ctype == 'text/plain':
            #print("text/plain loop")
            #print("Charset: {}".format(part.get_content_charset()))
            try:
                body_text = part.get_payload(decode=True).decode(part.get_content_charset())
            except Exception as e:
                logger.warning('Exception: {}'.format(e))
                body_text=part.get_payload(decode=False)  # decode
            #body_text = part.get_payload(decode=False)
            #print("\n\n\n")
            #print("body_text_______________")
            #print(body_text)
            #print("\n\n\n")
            test_text=1
        if ctype == 'text/html':
            #print("text/html loop")
            #print("Charset: {}".format(part.get_content_charset()))
            try:
                html=part.get_payload(decode=True).decode(part.get_content_charset())
            except Exception as e:
                logger.warning('Exception: {}'.format(e))
                html=part.get_payload(decode=False)
            #html=part.get_payload(decode=False)
            html=css_regex.sub('',str(html))
            soup=BeautifulSoup(html,'html.parser')
            if body_text=='':
                #print("\n\n\n")
                body_text=soup.text
                body_text=hex_regex.sub('',body_text)
                #print("body_text_______________")
                #print("\n\n\n")
                #body_text=body_text.decode()
            #print("\n\n\n")
            #print("body_HTML_______________")
            body_html=str(soup)
            #print(body_html)
            #print("\n\n\n")
            #if hex_regex.search(body):
            #   hex_log.write(filepath+"\n")
            text_Html=1
        else:
            body_text=part.get_payload(decode=True).decode("utf-8")

    return body_text, body_html, text_Html, test_text, num_attachment, content_disposition_list, content_type_list, Content_Transfer_Encoding_list, file_extension_list, charset_list, size_in_Bytes


############################

def read_corpus(path):

  # assumes a flat directory structure 
    files = filter(lambda x: x.endswith('.txt'), os.listdir(path))
    paths = map(lambda x: os.path.join(path,x), files)
    return list(paths)


############################


def get_url(body):
    url_regex=re.compile('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',flags=re.IGNORECASE|re.MULTILINE)
    url=re.findall(url_regex,body)
    return url
    #if url==[]:
    #    return url
    #else:
    #    url=re.findall(url_regex,body)
    #    return url


def On_the_Character_of_Phishing_URLs(url):
    #            a       b       c       d       e       f       g       h       i       j       k       l
    # m     n       o       p       q       r       s       t       u       v       w       x        y       z
    char_dist = [.08167, .01492, .02782, .04253, .12702, .02228, .02015, .06094, .06966, .00153, .00772, .04025, .02406,
                 .06749, .07507, .01929, .00095, .05987, .06327, .09056, .02758, .00978, .02360, .00150, .01974, .00074]

    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))

    url_char_dist = []
    for x in range(26):
        url_char_dist.append(url.count(chr(x + ord('a'))) / (count(url, string.ascii_letters)))

    #print(char_dist)
    #print(url_char_dist)

    # kolmogorov-shmirnov

    ks = stats.ks_2samp(url_char_dist, char_dist)
    #print("KS")
    #print(ks)

    # Kullback-Leibler Divergence

    kl = stats.entropy(url_char_dist, char_dist)
    #print("KL")
    #print(kl)

    # Euclidean dist

    ed = distance.euclidean(url_char_dist, char_dist)
    #print("ED")
    #print(ed)

    # Normalized Character Frequencies
    # this is just url_char_dist

    # Edit Distance
    # To Do
    edit_distance = 0

    # Length of Domain
    # done elsewhere

    # @ and - symbols
    # done elsewhere

    ## punctuation
    num_punct = count(url, string.punctuation)
    #print("# punct")
    #print(num_punct)

    ## TLDs
    tlds = ['.com', '.net', '.co.uk', '.jp', '.ru', '.org']  # add more?

    # Target words
    target_words = 0

    # IP Addr

    ## Suspicious words
    suspicious_words = 0

    return url_char_dist, ks[1], kl, ed, num_punct

def Behind_Phishing_Modi_Operendi_Features(url):
    url_length = len(url)
    parsed_url = urlparse(url)
    domain = '{uri.scheme}://{uri.hostname}/'.format(uri=parsed_url)
    domain_length = len(domain)

    lc_domain = domain.lower()

    letter_occ = []
    for x in range(26):
        letter_occ.append(lc_domain.count(chr(x + ord('a'))))

    # brands in url
    brands = []

    return url_length, domain_length, letter_occ

def my_isIPAddr(url):
    parsed_url = urlparse(url)
    domain = '{uri.hostname}'.format(uri=parsed_url)
    if re.match("^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) == None:
        return False
    return True



###########################
########################### Functions for features_url.py

def dns_ip_retrieve(dns_input_file):
    """
    Reads the the dns information from the input file
    dns_input_file: dump file created by the download_url.py
    return: list of dns entries and ip addresses
    """
    dns_entry = {}
    IPs = []
    dns = True
    for line in dns_input_file:
        cleaned = line.strip().rstrip('\n')
        if cleaned == '' or cleaned == 'DNS Lookup' or cleaned == 'IP':
            pass
        if dns:
            if line.startswith("++++++"):
                dns = False
            else:
                if line.split(" : ")[0] not in dns_entry:
                    ids = []
                    dns_entry[line.split(" : ")[0]] = ids

                dns_entry[line.split(" : ")[0]].append(line.split(" : ")[1])
        else:
            IPs = [e for e in cleaned.split(',')]
    return dns_entry, IPs


def my_isIPAddr(url):
    parsed_url = urlparse(url)
    domain = '{uri.hostname}'.format(uri=parsed_url)
    if re.match("^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) == None:
        return 0
    return 1

def mean_scaling(Array_Features):
    scaler = preprocessing.StandardScaler().fit(Array_Features)
    #get the mean
    mean_Array_Features=scaler.mean_
    #std_Array=scaler.std_ # does not exist
    #return scaled feature array:
    scaled_Array_Features=scaler.transform(Array_Features)
    return scaled_Array_Features

def min_max_scaling(Array_Features):
    min_max_scaler = preprocessing.MinMaxScaler()
    if np.shape(Array_Features)[0] == 1:
        minmaxScaled_Array_features=min_max_scaler.fit_transform(np.transpose(Array_Features))
        minmaxScaled_Array_features = np.transpose(minmaxScaled_Array_features)
    else:
        minmaxScaled_Array_features=min_max_scaler.fit_transform(Array_Features)
    #get min and max of array features
    min_Array_Features=min_max_scaler.min_
    #max_Array_Features=min_max_scaler.max_  #does not exit
    return minmaxScaled_Array_features

def abs_scaler(Array_Features):
    max_abs_scaler = preprocessing.MaxAbsScaler()
    maxabs_Array_features = max_abs_scaler.fit_transform(Array_Features)
    return maxabs_Array_features
def normalizer(Array_Features):
    Array_Features_normalized = preprocessing.normalize(Array_Features, norm='l2')
    return Array_Features_normalized

def Preprocessing(X):
    #summary=open(config["Summary"]["Path"],'w')
    summary=Features.summary
    summary.write("\n\n###### List of Preprocessing steps:\n")
    #Array_Features=Sparse_Matrix_Features.toarray()
    X_array=X.toarray()
    #X_test_array=X_test.toarray()
    #scaled_Array_Features=preprocessing.scale(Sparse_Matrix_Features) # Center data with the mean and then scale it using the std deviation
    #other method that keeps the model for testing
    #if config["Preprocessing"]["mean_scaling"] == "True":
    #    X_train=mean_scaling(X_train)
    #    X_test=mean_scaling(X_test)
    #    summary.write("\n Scaling using the mean.\n")
    #    print("Preprocessing: Mean_scaling")
    #    return X_train, X_test
    #    # return the scaler for testing data
    #    # Use min max to scale data because it's robust to very small standard deviations of features and preserving zero
    if config["Preprocessing"]["min_max_scaling"] == "True":
        X=min_max_scaling(X_array)
        #X_test=min_max_scaling(X_test_array)
        summary.write("\n Scaling using the min and max.\n")
        logger.info("Preprocessing: min_max_scaling")
        return X
        # use abs value to scale
    #elif config["Preprocessing"]["abs_scaler"] == "True":
    #    X_train=abs_scaler(X_train)
    #    X_test=abs_scaler(X_test)
    #    summary.write("\n Scaling using the absolute value.\n")
    #    print("Preprocessing: abs_scaler")
    #    return X_train, X_test
    #    #normalize the data???
    #elif config["Preprocessing"]["normalize"] == "True":
    #    X_train = normalizer(X_train)
    #    X_test = normalizer(X_test)
    #    summary.write("\n Normalizing.\n")
    #    print("Preprocessing: Normalizing")
    #    return X_train, X_test
    else:
        return X
    #return scaler, scaled_Array_Features, mean_Array_Features, min_Array_Features #, max_Array_Features
def Cleaning(dict1):
    count=0
    for item in dict1:
        #print(item)
        for key in item.keys():
            #print(item[key])
            #if item[key] == "None" or item[key] == "N/A" or item[key] == "Nan" :
            if item[key] in ["None", "N/A" ,"NaN", None]:
                original=item[key]
                item[key]= -1
                count+=1
                logger.debug("Value of {} changed from {} to {}".format(key,original,item[key]))


#list_id=list(range(1,len(list_features)+1))
#dict_features=dict(zip(list_id,list_features))
#print(list_features)
#print(list_time)

def Vectorization_Training(list_dict_features_train):
    vec=DictVectorizer()
    vec.fit(list_dict_features_train)
    sparse_matrix_features_train=vec.transform(list_dict_features_train)
    return sparse_matrix_features_train, vec

def Vectorization_Testing(list_dict_features_test, vec):
    sparse_matrix_features_test=vec.transform(list_dict_features_test)
    return sparse_matrix_features_test


def dump_features(header, content, list_features, features_output, list_dict,list_time, time_dict):
    logger.debug("list_features: " + str(len(list_features)))
    list_dict.append(copy.copy(list_features))
    time_dict.append(copy.copy(list_time))
    with open(features_output+"_feature_vector.pkl",'ab') as feature_tracking:
        pickle.dump("URL: "+header, feature_tracking)
        pickle.dump(list_features,feature_tracking)
    with open(features_output+"_html_content.pkl",'ab') as feature_tracking:
        pickle.dump("URL: "+header, feature_tracking)
        pickle.dump(content,feature_tracking)
    with open(features_output+"_feature_vector.txt",'a+') as f:
        f.write("URL: "+str(header) + '\n' + str(list_features).replace('{','').replace('}','').replace(': ',':').replace(',','') + '\n\n')
    with open(features_output+"_time_stats.txt",'a+') as f:
        f.write("URL: "+str(header) + '\n' + str(list_time).replace('{','').replace('}','').replace(': ',':').replace(',','') + '\n\n')

def dump_features_emails(header, list_features, features_output, list_dict,list_time, time_dict):
    logger.debug("list_features: " + str(len(list_features)))
    list_dict.append(copy.copy(list_features))
    time_dict.append(copy.copy(list_time))
    with open(features_output+"_feature_vector.pkl",'ab') as feature_tracking:
        pickle.dump("email: "+str(header), feature_tracking)
        pickle.dump(list_features,feature_tracking)
    with open(features_output+"_feature_vector.txt",'a+') as f:
        f.write("email: "+str(header) + '\n' + str(list_features).replace('{','').replace('}','').replace(': ',':').replace(',','') + '\n\n')
    with open(features_output+"_time_stats.txt",'a+') as f:
        f.write("email: "+str(header) + '\n' + str(list_time).replace('{','').replace('}','').replace(': ',':').replace(',','') + '\n\n')


def single_network_features(dns_info, IPS, IP_whois, whois_info, url, list_features, list_time):
    if config["Network_Features"]["network_features"] == "True":
        Features.Network_creation_date(whois_info, list_features, list_time)
        logger.debug("creation_date")

        Features.Network_expiration_date(whois_info, list_features, list_time)
        logger.debug("expiration_date")

        Features.Network_updated_date(whois_info, list_features, list_time)
        logger.debug("updated_date")

        Features.Network_as_number(IP_whois, list_features, list_time)
        logger.debug("as_number")

        Features.Network_number_name_server(dns_info, list_features, list_time)
        logger.debug("number_name_server")

        Features.Network_dns_ttl(url, list_features, list_time)
        logger.debug("dns_ttl")

        Features.Network_DNS_Info_Exists(url, list_features, list_time)
        logger.debug('DNS_Info_Exists')

def single_javascript_features(soup, html, list_features, list_time):
    if config["HTML_Features"]["HTML_features"] == "True" and config["Javascript_Features"]["javascript_features"] == "True":
        Features.Javascript_number_of_exec(soup, list_features, list_time)
        logger.debug("number_of_exec")

        Features.Javascript_number_of_escape(soup, list_features, list_time)
        logger.debug("number_of_escape")

        Features.Javascript_number_of_eval(soup, list_features, list_time)
        logger.debug("number_of_eval")

        Features.Javascript_number_of_link(soup, list_features, list_time)
        logger.debug("number_of_link")

        Features.Javascript_number_of_unescape(soup, list_features, list_time)
        logger.debug("number_of_unescape")

        Features.Javascript_number_of_search(soup, list_features, list_time)
        logger.debug("number_of_search")

        Features.Javascript_number_of_setTimeout(soup, list_features, list_time)
        logger.debug("number_of_setTimeout")

        Features.Javascript_number_of_iframes_in_script(soup, list_features, list_time)
        logger.debug("number_of_iframes_in_script")

        Features.Javascript_number_of_event_attachment(soup, list_features, list_time)
        logger.debug("number_of_event_attachment")

        Features.Javascript_rightclick_disabled(html, list_features, list_time)
        logger.debug("rightclick_disabled")

        Features.Javascript_number_of_total_suspicious_features(list_features,list_time)
        logger.debug("number_of_total_suspicious_features")

def single_url_feature(url, list_features,list_time):
    if config["URL_Features"]["url_features"] == "True":
        Features.URL_url_length(url, list_features, list_time)
        logger.debug("url_length")

        Features.URL_domain_length(url, list_features, list_time)
        logger.debug("domain_length")

        Features.URL_char_distance(url, list_features, list_time)
        logger.debug("url_char_distance")

        Features.URL_kolmogorov_shmirnov(list_features, list_time)
        logger.debug("kolmogorov_shmirnov")

        Features.URL_Kullback_Leibler_Divergence(list_features, list_time)
        logger.debug("Kullback_Leibler_Divergence")

        Features.URL_english_frequency_distance(list_features, list_time)
        logger.debug("english_frequency_distance")

        Features.URL_num_punctuation(url, list_features, list_time)
        logger.debug("num_punctuation")

        Features.URL_has_port(url, list_features, list_time)
        logger.debug("has_port")

        Features.URL_has_https(url, list_features, list_time)
        logger.debug("has_https")

        Features.URL_number_of_digits(url, list_features, list_time)
        logger.debug("number_of_digits")

        Features.URL_number_of_dots(url, list_features, list_time)
        logger.debug("number_of_dots")

        Features.URL_number_of_slashes(url, list_features, list_time)
        logger.debug("number_of_slashes")

        Features.URL_digit_letter_ratio(url, list_features, list_time)
        logger.debug("digit_letter_ratio")

        Features.URL_special_char_count(url, list_features, list_time)
        logger.debug("special_char_count")

        Features.URL_Top_level_domain(url, list_features, list_time)
        logger.debug("Top_level_domain")

        Features.URL_number_of_dashes(url, list_features, list_time)
        logger.debug('URL_number_of_dashes')

        Features.URL_Http_middle_of_URL(url, list_features, list_time)
        logger.debug('URL_Http_middle_of_URL')

        Features.URL_Has_More_than_3_dots(url, list_features, list_time)
        logger.debug('URL_Has_More_than_3_dots')

        Features.URL_Has_at_symbole(url, list_features, list_time)
        logger.debug("URL_Has_at_symbole")

        Features.URL_Has_anchor_tag(url, list_features, list_time)
        logger.debug("URL_Has_anchor_tag")

        Features.URL_Null_in_Domain(url, list_features, list_time)
        logger.debug("URL_Null_in_Domain")

        Features.URL_Token_Count(url, list_features, list_time)
        logger.debug("URL_Token_Count")

        Features.URL_Average_Path_Token_Length(url, list_features, list_time)
        logger.debug("URL_Average_Path_Token_Length")

        Features.URL_Average_Domain_Token_Length(url, list_features, list_time)
        logger.debug("URL_Average_Domain_Token_Length")

        Features.URL_Longest_Domain_Token(url, list_features, list_time)
        logger.debug('URL_Longest_Domain_Token')

        Features.URL_Protocol_Port_Match(url, list_features, list_time)
        logger.debug('URL_Protocol_Port_Match')

        Features.URL_Has_WWW_in_Middle(url, list_features, list_time)
        logger.debug('URL_Has_WWW_in_Middle')

        Features.URL_Has_Hex_Characters(url, list_features, list_time)
        logger.debug('URL_Has_Hex_Characters')

        Features.URL_Double_Slashes_Not_Beginning_Count(url, list_features, list_time)
        logger.debug("URL_Double_Slashes_Not_Beginning_Count")

        Features.URL_Brand_In_Url(url, list_features, list_time)
        logger.debug("URL_Bran_In_URL")

        Features.URL_Is_Whitelisted(url, list_features, list_time)
        logger.debug("URL_Is_Whitelisted")

def single_html_features(soup, html, url, list_features, list_time):
    if config["HTML_Features"]["html_features"] == "True":
        Features.HTML_number_of_tags(soup, list_features, list_time)
        logger.debug("number_of_tags")

        Features.HTML_number_of_head(soup, list_features, list_time)
        logger.debug("number_of_head")

        Features.HTML_number_of_html(soup, list_features, list_time)
        logger.debug("number_of_html")

        Features.HTML_number_of_body(soup, list_features, list_time)
        logger.debug("number_of_body")

        Features.HTML_number_of_titles(soup, list_features, list_time)
        logger.debug("number_of_titles")

        Features.HTML_number_suspicious_content(soup, list_features, list_time)
        logger.debug("number_suspicious_content")

        Features.HTML_number_of_iframes(soup, list_features, list_time)
        logger.debug("number_of_iframes")

        Features.HTML_number_of_input(soup, list_features, list_time)
        logger.debug("number_of_input")

        Features.HTML_number_of_img(soup, list_features, list_time)
        logger.debug("number_of_img")

        Features.HTML_number_of_tags(soup, list_features, list_time)
        logger.debug("number_of_tags")

        Features.HTML_number_of_scripts(soup, list_features, list_time)
        logger.debug("number_of_scripts")

        Features.HTML_number_of_anchor(soup, list_features, list_time)
        logger.debug("number_of_anchor")

        Features.HTML_number_of_video(soup, list_features, list_time)
        logger.debug("number_of_video")

        Features.HTML_number_of_audio(soup, list_features, list_time)
        logger.debug("number_of_audio")

        Features.HTML_number_of_hidden_iframe(soup, list_features, list_time)
        logger.debug("number_of_hidden_iframe")

        Features.HTML_number_of_hidden_div(soup, list_features, list_time)
        logger.debug("number_of_hidden_div")

        Features.HTML_number_of_hidden_object(soup, list_features, list_time)
        logger.debug("number_of_hidden_object")

        Features.HTML_number_of_hidden_iframe(soup, list_features, list_time)
        logger.debug("number_of_hidden_iframe")

        Features.HTML_inbound_count(soup, url, list_features, list_time)
        logger.debug("inbound_count")

        Features.HTML_outbound_count(soup, url, list_features, list_time)
        logger.debug("outbound_count")

        Features.HTML_inbound_href_count(soup, url, list_features, list_time)
        logger.debug("inbound_href_count")

        Features.HTML_outbound_href_count(soup, url, list_features, list_time)
        logger.debug("outbound_href_count")

        Features.HTML_Website_content_type(html, list_features, list_time)
        logger.debug("content_type")

        Features.HTML_content_length(html, list_features, list_time)
        logger.debug("content_length")

        Features.HTML_x_powered_by(html, list_features, list_time)
        logger.debug("x_powered_by")

        Features.HTML_URL_Is_Redirect(html, url, list_features, list_time)
        logger.debug("URL_Is_Redirect")

        Features.HTML_Is_Login(html.html, url, list_features, list_time)
        logger.debug("HTML_Is_Login")


def single_email_features(body_text, body_html, text_Html, test_text, num_attachment, content_disposition_list, content_type_list
                , Content_Transfer_Encoding_list, file_extension_list, charset_list, size_in_Bytes, subject, sender_full, recipient_full, recipient_name, recipient_full_address, recipient_domain,message_id
                , sender_name,sender_full_address,sender_domain,return_addr,x_virus_scanned,x_spam_flag,x_originating_ip, x_mailer
                , x_originating_hostname, dkim_signature, received_spf, x_original_authentication_results, authentication_results
                , received, Cc, Bcc, To, MIME_version, list_features, list_time):
    if config["Email_Features"]["extract header features"]=="True":
        Features.Email_Header_return_path(return_addr, list_features, list_time)
        logger.debug("return_path")
        Features.Email_Header_X_mailer(x_mailer,list_features, list_time)
        logger.debug("X_mailer")
        Features.Email_Header_X_originating_hostname(x_originating_hostname, list_features, list_time)
        logger.debug("X_originating_hostname")
        Features.Email_Header_X_originating_ip(x_originating_ip, list_features, list_time)
        logger.debug("X_originating_ip")
        Features.Email_Header_X_spam_flag(x_spam_flag, list_features, list_time)
        logger.debug("X_spam_flag")
        Features.Email_Header_X_virus_scanned(x_virus_scanned, list_features, list_time)
        logger.debug("X_virus_scanned")
        Features.Email_Header_X_Origininal_Authentication_results(x_original_authentication_results, list_features, list_time)
        logger.debug("X_Origininal_Authentication_results")
        Features.Email_Header_Received_SPF(received_spf, list_features, list_time)
        logger.debug("Received-SPF")
        Features.Email_Header_Dkim_Signature_Exists(dkim_signature, list_features, list_time)
        logger.debug("Dkim_Signature")
        Features.Email_Header_number_of_words_subject(subject, list_features, list_time)
        logger.debug("number_of_words_subject")
        Features.Email_Header_number_of_characters_subject(subject, list_features, list_time)
        logger.debug("number_of_characters_subject")
        Features.Email_Header_number_of_special_characters_subject(subject, list_features, list_time)
        logger.debug("numer_of_special_characters_subject")
        Features.Email_Header_binary_fwd(subject, list_features, list_time)
        logger.debug("binary_fwd")
        Features.Email_Header_vocab_richness_subject(subject, list_features, list_time)
        logger.debug("vocab_richness_subject")
        Features.Email_Header_compare_sender_return(sender_full_address, return_addr, list_features, list_time)
        logger.debug("compare_sender_return")
        Features.Email_Header_compare_sender_domain_message_id_domain(sender_domain , message_id, list_features, list_time)
        #Features.Content_Disposition(cdispo, list_features, list_time)
        #logger.debug("Content_Disposition")
        Features.Email_Header_Number_Cc(Cc, list_features, list_time)
        logger.debug("Number_Cc")
        Features.Email_Header_Number_Bcc(Bcc, list_features, list_time)
        logger.debug("Number_Bcc")
        Features.Email_Header_Number_To(To, list_features, list_time)
        logger.debug("Number_To")
        Features.Email_Header_Num_Content_type(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_type")
        Features.Email_Header_Num_Charset(charset_list, list_features, list_time)
        logger.debug("Email_Header_Num_Charset")
        Features.Email_Header_Num_Unique_Charset(charset_list, list_features, list_time)
        logger.debug("Email_Header_Num_Unique_Charset")
        Features.Email_Header_MIME_Version(MIME_version, list_features, list_time)
        logger.debug("Email_Header_MIME_Version")
        Features.Email_Header_Num_Unique_Content_type(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Unique_Content_type")
        Features.Email_Header_Num_Unique_Content_Disposition(content_disposition_list, list_features, list_time)
        logger.debug("Email_Header_Num_Unique_Content_Disposition")
        Features.Email_Header_Num_Content_Disposition(content_disposition_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Disposition")
        Features.Email_Header_Num_Content_Type_text_plain(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_text_plain")
        Features.Email_Header_Num_Content_Type_text_html(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_text_html")
        Features.Email_Header_Num_Content_Disposition(content_disposition_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Disposition")
        Features.Email_Header_Num_Content_Type_text_plain(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_text_plain")
        Features.Email_Header_Num_Content_Type_text_html(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_text_html")
        Features.Email_Header_Num_Content_Type_Multipart_Encrypted(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Multipart_Encrypted")
        Features.Email_Header_Num_Content_Type_Multipart_Mixed(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Multipart_Mixed")
        Features.Email_Header_Num_Content_Type_Multipart_form_data(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Multipart_form_data")
        Features.Email_Header_Num_Content_Type_Multipart_byterange(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Multipart_byterange")
        Features.Email_Header_Num_Content_Type_Multipart_Parallel(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Multipart_Parallel")
        Features.Email_Header_Num_Content_Type_Multipart_Report(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Multipart_Report")
        Features.Email_Header_Num_Content_Type_Multipart_Alternative(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Multipart_Alternative")
        Features.Email_Header_Num_Content_Type_Multipart_Digest_Num(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Multipart_Digest_Num")
        Features.Email_Header_Num_Content_Type_Multipart_Signed_Num(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Multipart_Signed_Num")
        Features.Email_Header_Num_Content_Type_Multipart_X_Mixed_Replaced(content_type_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Multipart_X_Mixed_Replaced")
        Features.Email_Header_Num_Content_Type_Charset_us_ascii(charset_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Charset_us_ascii")
        Features.Email_Header_Num_Content_Type_Charset_utf_8(charset_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Charset_utf_8")
        Features.Email_Header_Num_Content_Type_Charset_utf_7(charset_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Charset_utf_7")
        Features.Email_Header_Num_Content_Type_Charset_gb2312(charset_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Charset_gb2312")
        Features.Email_Header_Num_Content_Type_Charset_shift_jis(charset_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Charset_shift_jis")
        Features.Email_Header_Num_Content_Type_Charset_koi(charset_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Charset_koi")
        Features.Email_Header_Num_Content_Type_Charset_iso2022_jp(charset_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Type_Charset_iso2022_jp")
        Features.Email_Header_Num_Attachment(num_attachment, list_features, list_time)
        logger.debug("Email_Header_Num_Attachment")
        Features.Email_Header_Num_Unique_Attachment_types(file_extension_list, list_features, list_time)
        logger.debug("Email_Header_Num_Unique_Attachment_types")
        Features.Email_Header_Num_Content_Transfer_Encoding(Content_Transfer_Encoding_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Transfer_Encoding")
        Features.Email_Header_Num_Unique_Content_Transfer_Encoding(Content_Transfer_Encoding_list, list_features, list_time)
        logger.debug("Email_Header_Num_Unique_Content_Transfer_Encoding")
        Features.Email_Header_Num_Content_Transfer_Encoding_7bit(Content_Transfer_Encoding_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Transfer_Encoding_7bit")
        Features.Email_Header_Num_Content_Transfer_Encoding_8bit(Content_Transfer_Encoding_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Transfer_Encoding_8bit")
        Features.Email_Header_Num_Content_Transfer_Encoding_binary(Content_Transfer_Encoding_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Transfer_Encoding_binary")
        Features.Email_Header_Num_Content_Transfer_Encoding_quoted_printable(Content_Transfer_Encoding_list, list_features, list_time)
        logger.debug("Email_Header_Num_Content_Transfer_Encoding_quoted_printable")
        Features.Email_Header_Num_Unique_Attachment_types(file_extension_list, list_features, list_time)
        logger.debug("Email_Header_Num_Unique_Attachment_types")
        Features.Email_Header_size_in_Bytes(size_in_Bytes ,list_features, list_time)
        logger.debug("Email_Header_size_in_Bytes")
        Features.Email_Header_Received_count(received, list_features, list_time)
        logger.debug("Received_count")
        Features.Email_Header_Authentication_Results_SPF_Pass(authentication_results, list_features, list_time)
        logger.debug("Authentication_Results_SPF_Pass")
        Features.Email_Header_Authentication_Results_DKIM_Pass(authentication_results, list_features, list_time)
        logger.debug("Authentication_Results_DKIM_Pass")
        Features.Email_Header_Test_Html(text_Html, list_features, list_time)
        logger.debug("Test_Html")
        Features.Email_Header_Test_Text(test_text, list_features, list_time)
        logger.debug("test_text")
        Features.Email_Header_blacklisted_words_subject(subject, list_features, list_time)
        logger.debug("Email_blacklisted_words_subject")

    if config["Email_Features"]["extract body features"]=="True":
        Features.Email_Body_flesh_read_score(body_text, list_features, list_time)
        logger.debug("flesh_read_score")
        Features.Email_Body_smog_index(body_text, list_features, list_time)
        logger.debug("smog_index")
        Features.Email_Body_flesh_kincaid_score(body_text, list_features, list_time)
        logger.debug("flesh_kincaid_score")
        Features.Email_Body_coleman_liau_index(body_text, list_features, list_time)
        logger.debug("coleman_liau_index")
        Features.Email_Body_automated_readability_index(body_text, list_features, list_time)
        logger.debug("automated_readability_index")
        Features.Email_Body_dale_chall_readability_score(body_text, list_features, list_time)
        logger.debug("dale_chall_readability_score")
        Features.Email_Body_difficult_words(body_text, list_features, list_time)
        logger.debug("difficult_words")
        Features.Email_Body_linsear_score(body_text, list_features, list_time)
        logger.debug("linsear_score")
        Features.Email_Body_gunning_fog(body_text, list_features, list_time)
        logger.debug("gunning_fog")
        #Features.html_in_body(body, list_features, list_time)
        #print("html_in_body")
        Features.Email_Body_number_of_words_body(body_text, list_features, list_time)
        logger.debug("number_of_words_body")
        Features.Email_Body_number_of_characters_body(body_text, list_features, list_time)
        logger.debug("number_of_characters_body")
        Features.Email_Body_number_of_special_characters_body(body_text, list_features, list_time)
        logger.debug("number_of_special_characters_body")
        Features.Email_Body_vocab_richness_body(body_text, list_features, list_time)
        logger.debug("vocab_richness_body")
        Features.Email_Body_number_of_html_tags_body(body_html, list_features, list_time)
        logger.debug("number_of_html_tags_body")
        Features.Email_Body_number_of_unique_words_body(body_text, list_features, list_time)
        logger.debug("number_unique_words_body")
        Features.Email_Body_number_unique_chars_body(body_text, list_features, list_time)
        logger.debug("number_unique_chars_body")
        Features.Email_Body_end_tag_count(body_html, list_features, list_time)
        logger.debug("end_tag_count")
        Features.Email_Body_open_tag_count(body_html, list_features, list_time)
        logger.debug("open_tag_count")
        Features.Email_Body_recipient_name_body(body_text,recipient_name, list_features, list_time)
        logger.debug("recipient_name_body")
        Features.Email_Body_on_mouse_over(body_html, list_features, list_time)
        logger.debug("on_mouse_over")
        Features.Email_Body_count_href_tag(body_html, list_features, list_time)
        logger.debug("count_href_tag")
        Features.Email_Body_Function_Words_Count(body_text, list_features, list_time)
        logger.debug("Email_Body_Function_Words_Count")
        Features.Email_Body_Number_Of_Img_Links(body_html, list_features, list_time)
        logger.debug("Email_Body_Number_Of_Img_Links")
        Features.Email_Body_blacklisted_words_body(body_text, list_features, list_time)
        logger.debug("Email_Body_blacklisted_words_body")
        Features.Email_Body_Number_Of_Scripts(body_html, list_features, list_time)
        logger.debug("Email_Number_Of_Scripts")



def email_url_features(url_All, sender_domain, list_features, list_time):
    if config["Email_Features"]["extract body features"]=="True":
        Features.Email_URL_Number_Url(url_All, list_features, list_time)
        logger.debug("Number_Url")
        Features.Email_URL_Number_Diff_Domain(url_All, list_features, list_time)
        logger.debug("Number_Diff_Domain")
        Features.Email_URL_Number_link_at(url_All, list_features, list_time)
        logger.debug("Number_link_at")
        Features.Email_URL_Number_link_sec_port(url_All, list_features, list_time)
        logger.debug("Number_link_sec_port")
    #
    #Features.Number_link_IP(url_All, list_features, list_time)
    #logger.debug(Number_link_IP)
    #Features.Number_link_HTTPS(url_All, list_features, list_time)
    #logger.debug(Number_link_HTTPS)
    #Features.Number_Domain_Diff_Sender(url_All, sender_domain, list_features, list_time)
    #print(Number_Domain_Diff_Sender)
    #Features.Number_Link_Text(url_All, list_features, list_time)
    #print(Number_Link_Text)
    #Features.Number_link_port_diff_8080(url_All, list_features, list_time)
    #print(Number_link_port_diff_8080)



# sys.setdefaultencoding('utf-8')

# get filename from path
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def url_features(filepath, list_features, list_dict, list_time, time_dict, corpus, Bad_URLs_List):
    times = []
    try:
        with open(filepath,'r', encoding = "ISO-8859-1") as f:
            for rawurl in f:
                rawurl = rawurl.strip().rstrip()
                try:
                    if not rawurl:
                        continue
                    logger.debug("rawurl:" + str(rawurl))
                    Features.summary.write("URL: {}".format(rawurl))
                    t0 = time.time()
                    html, content, Error = Download_url.download_url(rawurl, list_time)
                    IPs, ipwhois, whois_output, domain = Download_url.extract_whois(html.url, list_time)
                    dns_lookup = Download_url.extract_dns_info(html.url, list_time)
                    if Error == 1:
                        logger.warning("This URL has trouble being extracted and will not be considered for further processing:{}".format(rawurl))
                        Bad_URLs_List.append(rawurl)
                    else:
                        logger.debug("download_url >>>>>>>>> complete")
                        times.append(time.time() - t0)
                        # include https or http
                        url = rawurl.strip().rstrip('\n')
                        if content=='':
                            soup=''
                        soup = BeautifulSoup(content, 'html5lib')   #content=html.text
                        single_html_features(soup, html, url, list_features, list_time)
                        single_url_feature(url, list_features, list_time)
                        logger.debug("html_featuers & url_features >>>>>> complete")
                        single_javascript_features(soup,html, list_features, list_time)
                        logger.debug("html_features & url_features & Javascript feautures >>>>>> complete")
                        single_network_features(dns_lookup, IPs, ipwhois, whois_output, url, list_features, list_time)
                        features_output= "Data_Dump/URLs_Backup/"+'_'.join(ntpath.normpath(filepath).split('\\'))
                        if not os.path.exists("Data_Dump/URLs_Backup"):
                            os.makedirs("Data_Dump/URLs_Backup")
                        #with open(features_output+"_feature_vector.pkl",'ab') as feature_tracking:
                        #    pickle.dump("URL: "+rawurl, feature_tracking)
                        #    pickle.dump(list_features,feature_tracking)
                        # with open("Data_Dump/URLs_Training/"+path_leaf(filepath)+"_feature_vector.pkl",'rb') as feature_tracking:
                        #     for i in range(len(list_dict)+1):
                        #         logger.debug(pickle.load(feature_tracking))
                        dump_features(rawurl, str(soup), list_features, features_output, list_dict, list_time, time_dict)
                        #with open(features_output+"_html_content.pkl",'ab') as feature_tracking:
                        #    pickle.dump("URL: "+rawurl, feature_tracking)
                        #    pickle.dump(str(soup),feature_tracking)
                        corpus.append(str(soup))
                except Exception as e:
                    logger.warning(traceback.format_exc())
                    logger.warning(e)
                    logger.warning("This URL has trouble being extracted and will not be considered for further processing:{}".format(rawurl))
                    Bad_URLs_List.append(rawurl)
               
    except Exception as e:
        logger.warning("exception: " + str(e))
        logger.debug(traceback.format_exc())
    logger.info("Download time is: {}".format(sum(times)/len(times)))

def email_features(filepath, list_features, features_output, list_dict, list_time, time_dict, corpus):
    try:
        with open(filepath,'r', encoding = "ISO-8859-1") as f:
            email=f.read()
            body_text, body_html, text_Html, test_text, num_attachment, content_disposition_list, content_type_list, Content_Transfer_Encoding_list, file_extension_list, charset_list, size_in_Bytes = extract_body(email)
            logger.debug("extract_body >>>> Done")

            url_All=get_url(body_html)

            logger.debug("extract urls from body >>>> Done")

            (subject, sender_full, recipient_full, recipient_name, recipient_full_address, recipient_domain,message_id
                , sender_name,sender_full_address, sender_domain, return_addr, x_virus_scanned, x_spam_flag, x_originating_ip, x_mailer
                , x_originating_hostname, dkim_signature, received_spf, x_original_authentication_results, authentication_results
                   , received, Cc, Bcc, To , MIME_version )= extract_header_fields(email)

            logger.debug("extract_header_fields >>>> Done")
            #header=extract_header(email)
            single_email_features(body_text, body_html, text_Html, test_text, num_attachment, content_disposition_list, content_type_list
                , Content_Transfer_Encoding_list, file_extension_list, charset_list, size_in_Bytes, subject, sender_full, recipient_full, str(recipient_name), recipient_full_address, recipient_domain,message_id
                , sender_name,sender_full_address,sender_domain,return_addr,x_virus_scanned,x_spam_flag,x_originating_ip, x_mailer
                , x_originating_hostname, dkim_signature, received_spf, x_original_authentication_results, authentication_results
                , received, Cc, Bcc, To, MIME_version, list_features, list_time)
            logger.debug("Email features >>>>>>>>>>> Done")

            #email_url_features(url_All,sender_domain, list_features, list_time)
            #print("Email Url Features extracted")
            #print(list_features)
            corpus.append(body_text)
            '''
            if url==[]:
                link=''
                single_url_feature(link, list_features, list_time)
            else:
                for link in url:
                    print(link)
                    single_url_feature(link, list_features, list_time)
            print("URL features extracted")
            '''
            dump_features_emails(filepath, list_features, features_output, list_dict, list_time, time_dict)


    except Exception as e:
        logger.warning("exception: " + str(e))
