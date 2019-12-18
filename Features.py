import numpy as np
import re
import sys
import os, os.path
from itertools import groupby
import nltk
from lxml import html as lxml_html
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
from scipy import sparse
from scipy.spatial import distance
import time
import pandas as pd
from bs4 import BeautifulSoup
from slimit import ast
from slimit.parser import Parser
from slimit.visitors import nodevisitor
from urllib.parse import urlparse
import dns.resolver
import csv
import tldextract
from datetime import datetime
import whois
from cryptography.x509.general_name import IPAddress
import pickle
import json
from Features_Support import *
#import User_options
import Download_url
from sklearn.feature_extraction import DictVectorizer
#import timeit
from sklearn import preprocessing
#import base64
from sklearn.datasets import dump_svmlight_file
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack
from pathlib import Path
from urllib.parse import urlparse
from urllib import request
import configparser
import time
from sklearn.externals import joblib
#from collections import deque
import logging
import pickle

logger = logging.getLogger('root')

config=configparser.ConfigParser()
config.read('Config_file.ini')
summary=open(config["Summary"]["Path"],'w')
################



##### Email Features:
#### Header Features:
#def message_id_domain(message_id, list_features, list_time):
#    if config['Features']["message_id_domain"] == "True":
#    #if User_options.message_id_domain == "True":
#        start=time.time()
#        try:
#            if message_id!="None":
#                print("message_id not non")
#                print(message_id)
#                message_id_domain=message_id.split("@")[1]
#                list_features["message_id_domain"]=message_id_domain
#            else:
#                #print("else loop")
#                message_id_domain="None"
#                list_features["message_id_domain"]=message_id_domain
#        except Exception as e:
#            logger.warning("exception: " + str(e))
#            list_features["message_id_domain"]="None"
#            logger.warning("exception handled")
#        end=time.time()
#        ex_time=end-start
#        list_time["message_id_domain"]=ex_time
#
#def message_id_left_part(message_id, list_features, list_time):
#    if config['Features']["message_id_left_part"] == "True":
#        start=time.time()
#        try:
#            if message_id!="None":
#                message_id_left_part=message_id.split("@")[1]
#                list_features["message_id_left_part"]=message_id_left_part
#            else:
#                message_id_left_part="None"
#                list_features["message_id_left_part"]=message_id_left_part
#        except Exception as e:
#            logger.warning("exception: " + str(e))
#            list_features["message_id_left_part"]="None"
#            logger.warning("exception handled")
#        end=time.time()
#        ex_time=end-start
#        list_time["message_id_left_part"]=ex_time

#def recipient_name(recipient_name, list_features, list_time):
#    if config['Features']["recipient_name"] == "True":
#        start=time.time()
#        try:
#            print("recipient_name {}".format(recipient_name))
#            list_features["recipient_name"]=recipient_name
#        except Exception as e:
#            logger.warning("exception: " + str(e))
#            list_features["recipient_name"]="None"
#        end=time.time()
#        ex_time=end-start
#        list_time["recipient_name"]=ex_time


#def recipient_full_address(recipient_full_address, list_features, list_time):
#    if config["Features"]["recipient_full_address"] == "True":
#        start=time.time()
#        try:
#            #print("recipient_full_address type: {}".format(type(recipient_full_address)))
#            print("recipient_full_address >>> {}".format(recipient_full_address))
#            #for address in recipient_full_address:
#            #    list_features["recipient_full_address_" + str(recipient_full_address.index(address))]=adderss
#            list_features["recipient_full_address"]=str(recipient_full_address)
#        except Exception as e:
#            logger.warning("exception: " + str(e))
#            list_features["recipient_full_address"]="None"
#        end=time.time()
#        ex_time=end-start
#        list_time["recipient_full_address"]=ex_time

#def recipient_domain(recipient_domain, list_features, list_time):
#    if config["Features"]["recipient_domain"] == "True":
#        start=time.time()
#        try:
#            #print("recipient domain: {}".format(recipient_domain))
#            #print("recipient domain type: {}".format(type(recipient_domain)))
#            if recipient_domain=="None":
#                for i in range(3):
#                    list_features["recipient_domain_"+str(i+1)]="None"
#            if recipient_domain is list:
#                #print("recipient_domain is list")
#                for i in range(3):
#                    list_features["recipient_domain_"+str(i+1)]=str(recipient_domain[i])
#                #for domain in recipient_domain:
#                #    list_features["recipient_domain_" + str(recipient_domain.index(domain)+1)]=domain
#            else:
#               list_features["recipient_domain_1"]=str(recipient_domain)
#               for i in range(1,3):
#                    list_features["recipient_domain_"+str(i+1)]="None"
#                #list_features["recipient_domain"]=recipient_domain
#        except Exception as e:
#            logger.warning("exception: " + str(e))
#            print("recipient domain exception")
#            list_features["recipient_domain"]="None"
#        #print("recipient_domain >>> {}".format(list_features["recipient_domain"]))
#        end=time.time()
#        ex_time=end-start
#        list_time["recipient_domain"]=ex_time

#def sender_name(sender_name, list_features, list_time):
#    if config["Features"]["sender_name"] == "True":
#        start=time.time()
#        try:
#            list_features["sender_name"]=sender_name
#        except Exception as e:
#            logger.warning("exception: " + str(e))
#            list_features["sender_name"]="None"
#        print("sender_name >>> {}".format(sender_name))
#        end=time.time()
#        ex_time=end-start
#        list_time["sender_name"]=ex_time

#def sender_domain(sender_domain, list_features, list_time):
#    if config["Features"]["sender_domain"] == "True":
#        start=time.time()
#        try:
#            list_features["sender_domain"]=sender_domain
#        except Exception as e:
#            logger.warning("exception: " + str(e))
#            list_features["sender_domain"]="None"
#        print("sender_domain >>> {}".format(sender_domain))
#        end=time.time()
#        ex_time=end-start
#        list_time["sender_domain"]=ex_time
#
#def return_address(return_addr, list_features, list_time):
#    if config["Features"]["return_address"] == "True":
#        start=time.time()
#        try:
#            list_features["return_address"]=return_addr
#        except Exception as e:
#            logger.warning("exception: " + str(e))
#            list_features["return_address"]="None"
#        print("return_address >>>> {}".format(return_addr))
#        end=time.time()
#        ex_time=end-start
#        list_time["return_address"]=ex_time
def Email_Header_Num_Content_type(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_type"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_type"]=len(content_type_list)
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_type"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_type"]=ex_time

def Email_Header_Num_Charset(charset_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Charset"] == "True":
        start=time.time()
        try:
            list_features["Num_Charset"] = len(charset_list)
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Charset"] = -1
        end=time.time()
        ex_time=end-start
        list_time["Num_Charset"]=ex_time

def Email_Header_Num_Unique_Charset(charset_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Unique_Charset"] == "True":
        start=time.time()
        try:
            list_features["Num_Unique_Charset"] = len(set(charset_list))
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Unique_Charset"] = -1
        end=time.time()
        ex_time=end-start
        list_time["Num_Unique_Charset"]=ex_time


def Email_Header_MIME_Version(MIME_version, list_features, list_time):
    if config["Email_Header_Features"]["MIME_Version"] == "True":
        start=time.time()
        try:
            #list_features["Email_Header_MIME_Version"]=MIME_version
            #print("Mime_version: {}".format(MIME_version))
            if MIME_version != None:
                list_features["MIME_Version"]=MIME_version
            else:
                list_features["MIME_Version"]=0
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["MIME_Version"]="N/A"
        end=time.time()
        ex_time=end-start
        list_time["MIME_Version"]=ex_time

def Email_Header_Num_Unique_Content_type(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Unique_Content_type"] == "True":
        start=time.time()
        try:
            list_features["Num_Unique_Content_type"]=len(set(content_type_list))
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Unique_Content_type"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Unique_Content_type"]=ex_time

def Email_Header_Num_Unique_Content_Disposition(content_disposition_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Unique_Content_Disposition"] == "True":
        start=time.time()
        try:
            list_features["Num_Unique_Content_Disposition"]=len(set(content_disposition_list))
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Unique_Content_Disposition"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Unique_Content_Disposition"]=ex_time

def Email_Header_Num_Content_Disposition(content_disposition_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Disposition"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Disposition"]=len(content_disposition_list)
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Disposition"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Disposition"]=ex_time

def Email_Header_Num_Content_Type_text_plain(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_text_plain"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_text_plain"]=content_type_list.count("text/plain")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_text_plain"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_text_plain"]=ex_time


def Email_Header_Num_Content_Type_text_html(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_text_html"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_text_html"]=content_type_list.count("text/html")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_text_html"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_text_html"]=ex_time

def Email_Header_Num_Content_Type_Multipart_Encrypted(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Multipart_Encrypted"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Multipart_Encrypted"]=content_type_list.count("multipart/encrypted")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Multipart_Encrypted"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Multipart_Encrypted"]=ex_time

def Email_Header_Num_Content_Type_Multipart_Mixed(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Multipart_Mixed"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Multipart_Mixed"]=content_type_list.count("multipart/mixed")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Multipart_Mixed"] = -1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Multipart_Mixed"]=ex_time

def Email_Header_Num_Content_Type_Multipart_form_data(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Multipart_form_data"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Multipart_form_data"] = content_type_list.count("multipart/form-data")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Multipart_form_data"] = -1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Multipart_form_data"]=ex_time

def Email_Header_Num_Content_Type_Multipart_byterange(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Multipart_byterange"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Multipart_byterange"] = content_type_list.count("multipart/byterange")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Multipart_byterange"] = -1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Multipart_byterange"]=ex_time

def Email_Header_Num_Content_Type_Multipart_Parallel(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Multipart_Parallel"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Multipart_Parallel"] = content_type_list.count("multipart/parallel")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Multipart_Parallel"] = -1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Multipart_Parallel"]=ex_time

def Email_Header_Num_Content_Type_Multipart_Report(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Multipart_Report"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Multipart_Report"] = content_type_list.count("multipart/report")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Multipart_Report"] = -1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Multipart_Report"]=ex_time

def Email_Header_Num_Content_Type_Multipart_Alternative(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Multipart_Alternative"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Multipart_Alternative"] = content_type_list.count("multipart/alternative")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Multipart_Alternative"] = -1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Multipart_Alternative"]=ex_time

def Email_Header_Num_Content_Type_Multipart_Digest_Num(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Multipart_Digest_Num"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Multipart_Digest_Num"] = content_type_list.count("multipart/digest")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Multipart_Digest_Num"] = -1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Multipart_Digest_Num"]=ex_time

def Email_Header_Num_Content_Type_Multipart_Signed_Num(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Multipart_Signed_Num"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Multipart_Signed_Num"] = content_type_list.count("multipart/signed")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Multipart_Signed_Num"] = -1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Multipart_Signed_Num"]=ex_time

def Email_Header_Num_Content_Type_Multipart_X_Mixed_Replaced(content_type_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Multipart_X_Mixed_Replaced"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Multipart_X_Mixed_Replaced"] = content_type_list.count("multipart/x-mixed-replaced")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Multipart_X_Mixed_Replaced"] = -1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Multipart_X_Mixed_Replaced"]=ex_time

def Email_Header_Num_Content_Type_Charset_us_ascii(charset_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Charset_us_ascii"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Charset_us_ascii"]=charset_list.count("us_ascii")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Charset_us_ascii"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Charset_us_ascii"]=ex_time

def Email_Header_Num_Content_Type_Charset_utf_8(charset_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Charset_utf_8"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Charset_utf_8"]=charset_list.count("utf_8")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Charset_utf_8"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Charset_utf_8"]=ex_time

def Email_Header_Num_Content_Type_Charset_utf_7(charset_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Charset_utf_7"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Charset_utf_7"]=charset_list.count("utf_7")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Charset_utf_7"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Charset_utf_7"]=ex_time

def Email_Header_Num_Content_Type_Charset_gb2312(charset_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Charset_gb2312"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Charset_gb2312"]=charset_list.count("gb2312")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Charset_gb2312"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Charset_gb2312"]=ex_time

def Email_Header_Num_Content_Type_Charset_shift_jis(charset_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Charset_shift_jis"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Charset_shift_jis"]=charset_list.count("shit_jis")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Charset_shift_jis"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Charset_shift_jis"]=ex_time

def Email_Header_Num_Content_Type_Charset_koi(charset_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Charset_koi"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Charset_koi"]=charset_list.count("koi")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Charset_koi"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Charset_koi"]=ex_time

def Email_Header_Num_Content_Type_Charset_iso2022_jp(charset_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Type_Charset_iso2022_jp"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Type_Charset_iso2022_jp"]=charset_list.count("iso2022-jp")
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Type_Charset_iso2022_jp"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Type_Charset_iso2022_jp"]=ex_time

def Email_Header_Num_Attachment(num_attachment, list_features, list_time):
    if config["Email_Header_Features"]["Num_Attachment"] == "True":
        start=time.time()
        try:
            list_features["Num_Attachment"]=num_attachment
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Attachment"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Attachment"]=ex_time

def Email_Header_Num_Unique_Attachment_types(file_extension_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Unique_Attachment_types"] == "True":
        start=time.time()
        try:
            list_features["Num_Unique_Attachment_types"]=len(set(file_extension_list))
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Unique_Attachment_types"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Unique_Attachment_types"]=ex_time

def Email_Header_Num_Content_Transfer_Encoding(Content_Transfer_Encoding_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Transfer_Encoding"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Transfer_Encoding"]=len(Content_Transfer_Encoding_list)
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Transfer_Encoding"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Transfer_Encoding"]=ex_time

def Email_Header_Num_Unique_Content_Transfer_Encoding(Content_Transfer_Encoding_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Unique_Content_Transfer_Encoding"] == "True":
        start=time.time()
        try:
            list_features["Num_Unique_Content_Transfer_Encoding"]=len(set(Content_Transfer_Encoding_list))
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Unique_Content_Transfer_Encoding"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Unique_Content_Transfer_Encoding"]=ex_time


def Email_Header_Num_Content_Transfer_Encoding_7bit(Content_Transfer_Encoding_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Transfer_Encoding_7bit"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Transfer_Encoding_7bit"]=Content_Transfer_Encoding_list.count('7bit')
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Transfer_Encoding_7bit"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Transfer_Encoding_7bit"]=ex_time

def Email_Header_Num_Content_Transfer_Encoding_8bit(Content_Transfer_Encoding_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Transfer_Encoding_8bit"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Transfer_Encoding_8bit"]=Content_Transfer_Encoding_list.count('8bit')
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Transfer_Encoding_8bit"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Transfer_Encoding_8bit"]=ex_time

def Email_Header_Num_Content_Transfer_Encoding_binary(Content_Transfer_Encoding_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Transfer_Encoding_binary"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Transfer_Encoding_binary"]=Content_Transfer_Encoding_list.count('binary')
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Transfer_Encoding_binary"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Transfer_Encoding_binary"]=ex_time

def Email_Header_Num_Content_Transfer_Encoding_quoted_printable(Content_Transfer_Encoding_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Content_Transfer_Encoding_quoted_printable"] == "True":
        start=time.time()
        try:
            list_features["Num_Content_Transfer_Encoding_quoted_printable"]=Content_Transfer_Encoding_list.count('quoted-printable')
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Content_Transfer_Encoding_quoted_printable"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Content_Transfer_Encoding_quoted_printable"]=ex_time

def Email_Header_Num_Unique_Attachment_types(file_extension_list, list_features, list_time):
    if config["Email_Header_Features"]["Num_Unique_Attachment_types"] == "True":
        start=time.time()
        try:
            list_features["Num_Unique_Attachment_types"]=len(set(file_extension_list))
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Num_Unique_Attachment_types"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Num_Unique_Attachment_types"]=ex_time

def Email_Header_size_in_Bytes(size_in_bytes ,list_features, list_time):
    if config["Email_Header_Features"]["size_in_Bytes"] == "True":
        start=time.time()
        try:
            list_features["size_in_Bytes"]=size_in_bytes
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["size_in_Bytes"]=-1
        end=time.time()
        ex_time=end-start
        list_time["size_in_Bytes"]=ex_time

def Email_Header_return_path(return_addr, list_features, list_time):
    if config["Email_Header_Features"]["return_path"] == "True":
        start=time.time()
        try:
            list_features["return_path"]=return_addr
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["return_path"]=-1
        end=time.time()
        ex_time=end-start
        list_time["return_path"]=ex_time

def Email_Header_X_mailer(x_mailer,list_features, list_time):
    if config["Email_Header_Features"]["X_mailer"] == "True":
        start=time.time()
        try:
            list_features["X_mailer"]=x_mailer
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["X_mailer"]=-1
        #print("x_mailer >> {}".format(x_mailer))
        end=time.time()
        ex_time=end-start
        list_time["X_mailer"]=ex_time

def Email_Header_X_originating_hostname(x_originating_hostname, list_features, list_time):
    if config["Email_Header_Features"]["X_originating_hostname"] == "True":
        start=time.time()
        try:
            list_features["X_originating_hostname"]=x_originating_hostname
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["X_originating_hostname"]=-1
        end=time.time()
        ex_time=end-start
        list_time["X_originating_hostname"]=ex_time

def Email_Header_X_originating_ip(x_originating_ip, list_features, list_time):
    if config["Email_Header_Features"]["X_originating_ip"] == "True":
        start=time.time()
        try:
            list_features["X_originating_ip"]=x_originating_ip
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["X_originating_ip"]=-1
        end=time.time()
        ex_time=end-start
        list_time["X_originating_ip"]=ex_time

def Email_Header_X_spam_flag(x_spam_flag, list_features, list_time):
    if config["Email_Header_Features"]["X_spam_flag"] == "True":
        start=time.time()
        try:
            list_features["X_Spam_flag"]=x_spam_flag
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["X_Spam_flag"]=-1
        end=time.time()
        ex_time=end-start
        list_time["X_Spam_flag"]=ex_time

def Email_Header_X_virus_scanned(x_virus_scanned, list_features, list_time):
    if config["Email_Header_Features"]["X_virus_scanned"] == "True":
        start=time.time()
        try:
            list_features["X_virus_scanned"]=x_virus_scanned
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["X_virus_scanned"]=-1
        end=time.time()
        ex_time=end-start
        list_time["X_virus_scanned"]=ex_time

def Email_Header_Received_count(received, list_features, list_time):
    if config["Email_Header_Features"]["Received_count"] == "True":
        start=time.time()
        #print("received {}".format(received))
        try:
            if received=="None":
                list_features["Received_count"]=0
            else:
                list_features["Received_count"]=len(received)
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Received_count"]=-1
        #print("Received count >>>> {}".format(received))
        end=time.time()
        ex_time=end-start              
        list_time["Received_count"]=ex_time


def Email_Header_Authentication_Results_SPF_Pass(authentication_results, list_features, list_time):
    if config["Email_Header_Features"]["Authentication_Results_SPF_Pass"] == "True":
        start=time.time()
        try:
            if "spf=pass" in authentication_results:
                list_features["Authentication_Results_SPF_Pass"]=1
            else:
                list_features["Authentication_Results_SPF_Pass"]=0
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Authentication_Results_SPF_Pass"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Authentication_Results_SPF_Pass"]=ex_time

def Email_Header_Authentication_Results_DKIM_Pass(authentication_results, list_features, list_time):
    if config["Email_Header_Features"]["Authentication_Results_DKIM_Pass"] == "True":
        start=time.time()
        try:
            if "dkim=pass" in authentication_results:
                list_features["Authentication_Results_DKIM_Pass"]=1
            else:
                list_features["Authentication_Results_DKIM_Pass"]=0
            #list_features["Authentication_Results"]=authentication_results
        except Exception as e:
            logger.warning("exception: " + str(e))
            #list_features["Authentication_Results"]="None"
            list_features["Authentication_Results_DKIM_Pass"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Authentication_Results_DKIM_Pass"]=ex_time

def Email_Header_X_Origininal_Authentication_results(x_original_authentication_results, list_features, list_time):
    if config["Email_Header_Features"]["X_Origininal_Authentication_results"] == "True":
        start=time.time()
        try:
            list_features["X_Origininal_Authentication_results"]=x_original_authentication_results
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["X_Origininal_Authentication_results"]=-1
        #print("X_Origininal_Authentication_results >>>> {}".format(x_original_authentication_results))
        end=time.time()
        ex_time=end-start
        list_time["X_Origininal_Authentication_results"]=ex_time

def Email_Header_Received_SPF(received_spf, list_features, list_time):
    if config["Email_Header_Features"]["Received_SPF"] == "True":
        start=time.time()
        try:
            list_features["Received_SPF"]=received_spf
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Received_SPF"]=-1
        #print("Received_SPF >>>> {}".format(received_spf))
        end=time.time()
        ex_time=end-start
        list_time["Received_SPF"]=ex_time

def Email_Header_Dkim_Signature_Exists(dkim_signature, list_features, list_time):
    if config["Email_Header_Features"]["Dkim_Signature_Exists"] == "True":
        start=time.time()
        try:
            #dkim_signature is boolean
            list_features["Dkim_Signature_Exists"]=dkim_signature
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Dkim_Signature_Exists"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Dkim_Signature_Exists"]=ex_time


def Email_Header_compare_sender_domain_message_id_domain(sender_domain , message_id, list_features, list_time):
    #global list_features
    if config["Email_Header_Features"]["compare_sender_domain_message_id_domain"] == "True":
        start=time.time()
        try:
            if message_id!="None":
                message_id_domain=message_id.split("@")[1]
            else:
                message_id_domain="None"
        except Exception as e:
            logger.warning("exception: " + str(e))
            message_id_domain="None"
        if message_id_domain != "None":
            compare_sender_domain_message_id_domain=int(bool(sender_domain==message_id_domain))
            list_features["compare_sender_domain_message_id_domain"]=compare_sender_domain_message_id_domain
        else:
            compare_sender_domain_message_id_domain=0
            list_features["compare_sender_domain_message_id_domain"]=compare_sender_domain_message_id_domain        
        #print("compare_sender_domain_message_id_domain >>>> {}".format(compare_sender_domain_message_id_domain))
        end=time.time()
        ex_time=end-start
        list_time["compare_sender_domain_message_id_domain"]=ex_time


def Email_Header_compare_sender_return(sender_full_address, return_addr, list_features, list_time):
    if config["Email_Header_Features"]["compare_sender_return"] == "True":
        start=time.time()
        try:
            compare_sender_return=int(bool(sender_full_address==return_addr))
        except Exception as e:
            logger.warning("exception: " + str(e))
            compare_sender_return=-1
        list_features["compare_sender_return"]=compare_sender_return    
        end=time.time()
        ex_time=end-start
        list_time["compare_sender_return"]=ex_time

def Email_Header_Test_Html(test_Html, list_features, list_time):
    if config["Email_Header_Features"]["Test_Html"] == "True":
        start=time.time()
        try:
            list_features["Test_Html"]=test_Html
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Test_Html"]=-1    
        end=time.time()
        ex_time=end-start
        list_time["Test_Html"]=ex_time

def Email_Header_Test_Text(test_text, list_features, list_time):
    if config["Email_Header_Features"]["Test_Text"] == "True":
        start=time.time()
        try:
            list_features["Test_Text"]=test_text
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Test_Text"]=-1    
        end=time.time()
        ex_time=end-start
        list_time["Test_Text"]=ex_time

##### Email URL features
def Email_Number_Url(url_All, list_features, list_time):
    if config["Email_URL_Features"]["Number_Url"] == "True":
        start=time.time()
        try:
            list_features["Number_Url"]=len(url_All)
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Number_Url"]=-1    
        end=time.time()
        ex_time=end-start
        list_time["Number_Url"]=ex_time

def Email_URL_Number_Diff_Domain(url_All, list_features, list_time):
    if config["Email_URL_Features"]["Number_Diff_Domain"] == "True":
        start=time.time()
        list_Domains=[]
        try:
            for url in url_All:
                parsed_url=urlparse(url)
                domain = parsed_url.hostname
                list_Domains.append(domain)
                #if domain not in list_Domains:
                #    list_Domains.append(domain)
            list_features["Number_Diff_Domain"]=len(set(list_Domains))
        except Exception as e:
            logger.warning("exception: "+str(e))
            list_features["Number_Diff_Domain"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Number_Diff_Domain"]=ex_time

def Email_URL_Number_Diff_Subdomain(url_All, list_features, list_time):
    if config["Email_URL_Features"]["Number_Diff_Subdomain"] == "True":
        start=time.time()
        list_Subdomains=[]
        try:
            for url in url_All:
                parsed_url=urlparse(url)
                domain = parsed_url.hostname
                subdomain=domain.split('.')[0]
                list_Subdomains.append(subdomain)
                #if domain not in list_Domains:
                #    list_Domains.append(domain)
            list_features["Number_Diff_Subdomain"]=len(set(list_Subdomains))
        except Exception as e:
            logger.warning("exception: "+str(e))
            list_features["Number_Diff_Subdomain"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Number_Diff_Subdomain"]=ex_time

def Email_URL_Number_link_at(url_All, list_features, list_time):
    if config["Email_URL_Features"]["Number_link_at"] == "True":
        start=time.time()
        count=0
        try:
            for url in url_All:
                if "@" in url:
                    count+=1
                    list_features["Number_link_at"]=count
        except Exception  as e:
            logger.warning("exception: " + str(e))
            list_features["Number_link_at"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Number_link_at"]=ex_time

def Email_URL_Number_link_sec_port(url_All, list_features, list_time):
    if config["Email_URL_Features"]["Number_link_sec_port"] == "True":
        start=time.time()
        count=0
        try:
            for url in url_All:
                if "::443" in url:
                    count+=1
                    list_features["Number_link_sec_port"]=count
        except Exception  as e:
            logger.warning("exception: " + str(e))
            list_features["Number_link_sec_port"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Number_link_sec_port"]=ex_time


#### Body Features:
def Email_Body_recipient_name_body(body,recipient_name, list_features, list_time):
    if config["Email_Body_Features"]["recipient_name_body"] == "True":
        start=time.time()
        try:
            recipient_name_body= int(bool(recipient_name in body))
        except Exception as e:
            logger.warning("exception: " + str(e))
            recipient_name_body=-1
        list_features["recipient_name_body"]=recipient_name_body
        end=time.time()
        ex_time=end-start
        list_time["compare_sender_return"]=ex_time

#def html_in_body(body, list_features, list_time):
#    if config["Features"]["html_in_body"] == "True":
#        start=time.time()
#        Email_Body_html=re.compile(r'text/html', flags=re.IGNORECASE)
#        try:
#            html_in_body=int(bool(re.search(Email_Body_html, body)))
#        except Exception as e:
#            logger.warning("exception: " + str(e))
#            html_in_body=0
#        list_features["html_in_body"]=html_in_body
#        end=time.time()
#        ex_time=end-start
#        list_time["html_in_body"]=ex_time
#        #list_features[""]=

def Email_Body_number_of_words_body(body, list_features, list_time):
    if config["Email_Body_Features"]["number_of_words_body"] == "True":
        start=time.time()
        try:
            number_of_words_body = len(re.findall(r'\w+', body))
        except Exception as e:
            logger.warning("exception: " + str(e))
            number_of_words_body = -1
        list_features["number_of_words_body"]=number_of_words_body
        end=time.time()
        ex_time=end-start
        list_time["number_of_words_body"]=ex_time

def Email_Body_number_of_unique_words_body(body, list_features, list_time):
    if config["Email_Body_Features"]["number_of_unique_words_body"] == "True":
        start=time.time()
        if body:
            try:
                number_of_words_body = len(set(re.findall(r'\w+', body)))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_words_body = -1
            list_features["number_of_unique_words_body"]=number_of_words_body
        else:
            list_features["number_of_unique_words_body"]=-1
        end=time.time()
        ex_time=end-start
        list_time["number_of_unique_words_body"]=ex_time

def Email_Body_number_of_characters_body(body, list_features, list_time):
    if config["Email_Body_Features"]["number_of_characters_body"] == "True":
        start=time.time()
        try:
            number_of_characters_body = len(re.findall(r'\w', body))
        except Exception as e:
            logger.warning("exception: " + str(e))
            number_of_characters_body =  -1
        list_features["number_of_characters_body"]=number_of_characters_body
        end=time.time()
        ex_time=end-start
        list_time["number_of_characters_body"]=ex_time

def Email_Body_number_of_special_characters_body(body, list_features, list_time):
    if config["Email_Body_Features"]["number_of_special_characters_body"] == "True":
        start=time.time()
        try:
            number_of_characters_body = len(re.findall(r'\w', body))
            number_of_special_characters_body = len(body)-number_of_characters_body-len(re.findall(r' ', body))
        except Exception as e:
            logger.warning("exception: " + str(e))
            number_of_special_characters_body = -1
        list_features["number_of_special_characters_body"]=number_of_special_characters_body
        end=time.time()
        ex_time=end-start
        list_time["number_of_special_characters_body"]=ex_time


def Email_Body_vocab_richness_body(body, list_features, list_time):
    if config["Email_Body_Features"]["vocab_richness_body"] == "True":
        start=time.time()
        try:
            vocab_richness_body=yule(body)
        except Exception as e:
            logger.warning("exception: " + str(e))
            vocab_richness_body = -1
        list_features["vocab_richness_body"]=vocab_richness_body
        end=time.time()
        ex_time=end-start
        list_time["vocab_richness_body"]=ex_time


def Email_Body_number_of_html_tags_body(body, list_features, list_time):
    if config["Email_Body_Features"]["number_of_html_tags_body"] == "True":
        start=time.time()
        try:
            if body:
                number_of_html_tags_body=len(re.findall(r'<.*>',body))
            else:
                number_of_html_tags_body=0
        except Exception as e:
            logger.warning("exception: " + str(e))
            number_of_html_tags_body= -1 
        list_features["number_of_html_tags_body"]=number_of_html_tags_body
        end=time.time()
        ex_time=end-start
        list_time["number_of_html_tags_body"]=ex_time

def Email_Body_number_unique_chars_body(body, list_features, list_time):
    if config["Email_Body_Features"]["number_unique_chars_body"] == "True":
        start=time.time()
        try:
            if body:
                number_unique_chars_body=len(set(body))-1
            else:
                number_unique_chars_body=0
        except Exception as e:
            logger.warning("exception: " + str(e))
            number_unique_chars_body = -1
        list_features["number_unique_chars_body"]=number_unique_chars_body
        end=time.time()
        ex_time=end-start
        list_time["number_unique_chars_body"]=ex_time
        #list_features[""]=

def Email_Body_greetings_body(body, list_features, list_time):
    if config["Email_Body_Features"]["greetings_body"] == "True":
        start=time.time()
        try:
            if body:
                dear_user=re.compile(r'Dear User', flags=re.IGNORECASE)
                greetings_body=int(bool(re.search(dear_user, body)))
            else:
                greetings_body=0
        except Exception as e:
            logger.warning("exception: " + str(e))
            greetings_body=-1
        list_features["greetings_body"]=greetings_body
        end=time.time()
        ex_time=end-start
        list_time["greetings_body"]=ex_time
        #list_features[""]=

def Email_Body_hidden_text(body, list_features, list_time):
    if config["Email_Body_Features"]["hidden_text"] == "True":
        start=time.time()
        regex_font_color=re.compile(r'<font +color="#FFFFF[0-9A-F]"',flags=re.DOTALL)
        try:
            if body:
                hidden_text=int(bool(regex_font_color.search(body)))
            else:
                hidden_text=0
        except Exception as e:
            logger.warning("exception: " + str(e))
            hidden_text=-1
        list_features["hidden_text"]=hidden_text
        end=time.time()
        ex_time=end-start
        list_time["hidden_text"]=ex_time

def Email_Body_count_href_tag(body, list_features, list_time):
    if config["Email_Body_Features"]["count_href_tag"] == "True":
        start=time.time()
        ultimate_regexp =re.compile(r"(?i)<\/?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)\/?>", flags=re.MULTILINE)
        count_href_tag=0
        try:
            if body:
                for match in re.finditer(ultimate_regexp,body):
                    if repr(match.group()).startswith("'<a"):
                        count_href_tag+=1
            else:
                count_href_tag=0
        except Exception as e:
            logger.warning("exception: " + str(e))
            count_href_tag=-1
        list_features["count_href_tag"]=count_href_tag
        end=time.time()
        ex_time=end-start
        list_time["count_href_tag"]=ex_time

def Email_Body_end_tag_count(body, list_features, list_time):
    if config["Email_Body_Features"]["end_tag_count"] == "True":
        start=time.time()
        ultimate_regexp =re.compile(r"(?i)<\/?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)\/?>", flags=re.MULTILINE)
        open_tag_count=0
        end_tag_count=0
        try:
            if body:
                for match in re.finditer(ultimate_regexp,body):
                    if repr(match.group()).startswith("'</"):
                        end_tag_count += 1
            else:
                end_tag_count=0
        except Exception as e:
            logger.warning("exception: " + str(e))
            end_tag_count=-1
        list_features["end_tag_count"]=end_tag_count
        end=time.time()
        ex_time=end-start
        list_time["end_tag_count"]=ex_time
        #list_features[""]=

def Email_Body_open_tag_count(body, list_features, list_time):
    if config["Email_Body_Features"]["open_tag_count"] == "True":
        start=time.time()
        ultimate_regexp =re.compile(r"(?i)<\/?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)\/?>", flags=re.MULTILINE)
        open_tag_count=0
        end_tag_count=0
        try:
            if body:
                for match in re.finditer(ultimate_regexp,body):
                    if repr(match.group()).startswith("'</"):
                        end_tag_count += 1
                    else:
                        open_tag_count += 1
            else:
                open_tag_count=0
        except Exception as e:
            logger.warning("exception: " + str(e))
            open_tag_count=-1
        list_features["open_tag_count"]=open_tag_count
        end=time.time()
        ex_time=end-start
        list_time["open_tag_count"]=ex_time

def Email_Body_on_mouse_over(body, list_features, list_time):
    if config["Email_Body_Features"]["on_mouse_over"] == "True":
        start=time.time()
        ultimate_regexp =re.compile(r"(?i)<\/?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)\/?>", flags=re.MULTILINE)
        on_mouse_over=0
        try:
            if body:
                for match in re.finditer(ultimate_regexp,body):
                    if repr(match.group()).startswith("'<a onmouseover"):
                        on_mouse_over += 1
            else:
                on_mouse_over=0
        except Exception as e:
            logger.warning("exception: " + str(e))
            on_mouse_over=-1
        list_features["on_mouse_over"]=on_mouse_over
        #list_features[""]=
        end=time.time()
        ex_time=end-start
        list_time["on_mouse_over"]=ex_time

def Email_Body_blacklisted_words_body(body, list_features, list_time):
    if config["Email_Body_Features"]["blacklisted_words_body"] == "True":
        start=time.time()
        blacklist_body=["urgent", "account", "closing", "act now", "click here", "limitied", "suspension", "your account", "verify your account", "agree", 'bank', 'dear'
                        ,"update", "comfirm", "customer", "client", "Suspend", "restrict", "verify", "login", "ssn", 'username','click','log','inconvenien','alert', 'paypal']        
        blacklist_body_count=[]
        if body:
            for word in blacklist_body:
                try:
                    word_count=len(re.findall(word,body.lower()))
                    #blakclist_body_count.append(word_count)
                    list_features[word+"_count_in_body"]=word_count
                except Exception as e:
                    logger.warning("exception: " + str(e))
                    list_features[word+"_count_in_body"]=-1
        else:
            for word in blacklist_body:
                list_features[word+"_count_in_body"]=0
        #list_features["blacklisted_words"]=blacklisted_words
        end=time.time()
        ex_time=end-start
        list_time["blacklisted_words_body"]=ex_time
        #list_features[""]=

def Email_Header_blacklisted_words_subject(subject, list_features, list_time):
    if config["Email_Header_Features"]["blacklisted_words_subject"] == "True":
        start=time.time()
        blacklist_subject=["urgent", "account", "closing", "act now", "click here", "limitied", "suspension", "your account", "verify your account", "agree", 'bank', 'dear'
                        ,"update", "comfirm", "customer", "client", "Suspend", "restrict", "verify", "login", "ssn", 'username','click','log','inconvenien','alert', 'paypal']        
        blacklist_subject_count=[]
        if subject:
            for word in blacklist_subject:
                try:
                    word_count=len(re.findall(word,subject.lower()))
                    list_features[word+"_count_in_subject"]=word_count
                except Exception as e:
                    logger.warning("exception: " + str(e))
                    list_features[word+"_count_in_subject"]=-1
        else:
            for word in blacklist_subject:
                list_features[word+"_count_in_subject"]=0
        end=time.time()
        ex_time=end-start
        list_time["blacklisted_words_subject"]=ex_time

def Email_Header_Number_Cc(cc, list_features, list_time):
    if config["Email_Header_Features"]["Number_Cc"] == "True":
        start=time.time()
        try:
            if cc:
                list_features["Number_Cc"]=len(cc)
            else:
                list_features["Number_Cc"]=0
        except Exception as e:
            logger.warning("exception: "+str(e))
            list_features["Number_Cc"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Number_Cc"]=ex_time

def Email_Header_Number_Bcc(Bcc, list_features, list_time):
    if config["Email_Header_Features"]["Number_Bcc"] == "True":
        start=time.time()
        try:
            if Bcc:
                list_features["Number_Bcc"]=len(Bcc)
            else:
                list_features["Number_Bcc"]=0
        except Exception as e:
            logger.warning("exception: " + str(e))
            list_features["Number_Bcc"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Number_Bcc"]=ex_time

def Email_Header_Number_To(To, list_features, list_time):
    if config["Email_Header_Features"]["Number_To"] == "True":
        start=time.time()
        try:
            if To:
                if len(To.split(',')) >= len(To.split(';')):
                    list_features["Number_To"]=len(To.split(','))
                else:
                    list_features["Number_To"]=len(To.split(';'))
            else:
                list_features["Number_To"]=0
        except Exception as e:
            logger.warning("exception: "+ str(e))
            list_features["Number_To"]=-1
        end=time.time()
        ex_time=end-start
        list_time["Number_To"]=ex_time

def Email_Body_Number_Of_Scripts(body, list_features, list_time):
    if config["Email_Body_Features"]["Number_Of_Scripts"] == "True":
        start=time.time()
        Number_Of_Scripts=0
        if body:
            soup = BeautifulSoup(body, "html.parser")
            try:
                 list_features["Number_Of_Scripts"]=len(soup.find_all('script'))
            except Exception as e:
                 logger.warning("exception :{}".format(e))
                 list_features["Number_Of_Scripts"]=-1
        else:
            list_features["Number_Of_Scripts"]=0
        end=time.time()
        ex_time=end-start
        list_time["Number_Of_Scripts"]=ex_time

def Email_Body_Number_Of_Img_Links(body, list_features, list_time):
    if config["Email_Body_Features"]["Number_Of_Img_Links"] == "True":
        start=time.time()
        Number_Of_Img_Links=0
        soup = BeautifulSoup(body, "html.parser")
        if body:
            try:
                 list_features["Number_Of_Img_Links"]=len(soup.find_all('img'))
            except Exception as e:
                 logger.warning("exception :{}".format(e))
                 list_features["Number_Of_Img_Links"]=-1
        else:
            list_features["Number_Of_Img_Links"]=0
        end=time.time()
        ex_time=end-start
        list_time["Number_Of_Img_Links"]=ex_time

def Email_Body_Function_Words_Count(body, list_features, list_time):
    if config["Email_Body_Features"]["Function_Words_Count"] == "True":
        start=time.time()
        Function_Words_Count=0
        if body:
            try:
                for word in body.split(' '):
                    if word in Function_words_list:
                        Function_Words_Count=+1   
            except Exception as e:
                logger.warning("exception: {}".format(e))
                Function_Words_Count = -1

        list_features["Function_Words_Count"]=Function_Words_Count
        end=time.time()
        ex_time=end-start
        list_time["Email_Body_Function_Words_Count"]=ex_time
# def  bodyTextNotSimSubjectAndMinOneLink()
# def Email_Body_body_num_func_words(body, list_features, list_time):
#   if config["Email_Body_Features"]["body_num_func_words"] == "True":
#       start=time.time()
#       body_num_func_words=0



#def body_unique_words() x
#def Email_Body_num_img_links() x
#def num_of_sub_domains() x
#def blacklist_words_in_subject() x


# source for style metrics: https://pypi.python.org/pypi/textstat
## Styles metrics:
def Email_Body_flesh_read_score(body, list_features, list_time):
    if config["Email_Body_Features"]["flesh_read_score"] == "True":
        start=time.time()
        if body:
            try:
                flesh_read_score=textstat.flesch_reading_ease(body)
            except Exception as e:
                logger.warning("exception: " + str(e))
                flesh_read_score=-1
        else:
            flesh_read_score=0
        list_features["flesh_read_score"]=flesh_read_score
        end=time.time()
        ex_time=end-start
        list_time["flesh_read_score"]=ex_time
        #list_features[""]=

def Email_Body_smog_index(body, list_features, list_time):
    if config["Email_Body_Features"]["smog_index"] == "True":
        start=time.time()
        if body:
            try:
                smog_index=textstat.smog_index(body)
            except Exception as e:
                logger.warning("exception: " + str(e))
                smog_index=-1
        else:
            smog_index=0
        list_features["smog_index"]=smog_index
        end=time.time()
        ex_time=end-start
        list_time["smog_index"]=ex_time
        #list_features[""]=

def Email_Body_flesh_kincaid_score(body, list_features, list_time):
    if config["Email_Body_Features"]["flesh_kincaid_score"] == "True":
        start=time.time()
        if body:
            try:
                flesh_kincaid_score=textstat.flesch_kincaid_grade(body)
            except Exception as e:
                logger.warning("exception: " + str(e))
                flesh_kincaid_score=-1
        else:
            flesh_kincaid_score=0
        list_features["flesh_kincaid_score"]=flesh_kincaid_score
        end=time.time()
        ex_time=end-start
        list_time["flesh_kincaid_score"]=ex_time
        #list_features[""]=

def Email_Body_coleman_liau_index(body, list_features, list_time):
    if config["Email_Body_Features"]["coleman_liau_index"] == "True":
        start=time.time()
        if body:    
            try:
                coleman_liau_index=textstat.coleman_liau_index(body)
            except Exception as e:
                logger.warning("exception: " + str(e))
                coleman_liau_index=-1
        else:
            coleman_liau_index=0
        list_features["coleman_liau_index"]=coleman_liau_index
        end=time.time()
        ex_time=end-start
        list_time["coleman_liau_index"]=ex_time
        #list_features[""]=

def Email_Body_automated_readability_index(body, list_features, list_time):
    if config["Email_Body_Features"]["automated_readability_index"] == "True":
        start=time.time()
        if body:    
            try:
                automated_readability_index=textstat.automated_readability_index(body)
            except Exception as e:
                logger.warning("exception: " + str(e))
                automated_readability_index=-1
        else:
            automated_readability_index=0
        list_features["automated_readability_index"]=automated_readability_index
        end=time.time()
        ex_time=end-start
        list_time["automated_readability_index"]=ex_time
        #list_features[""]=

def Email_Body_dale_chall_readability_score(body, list_features, list_time):
    if config["Email_Body_Features"]["dale_chall_readability_score"] == "True":
        start=time.time()
        if body:
            try:
                dale_chall_readability_score=textstat.dale_chall_readability_score(body)
            except Exception as e:
                logger.warning("exception: " + str(e))
                dale_chall_readability_score=-1
        else:
            dale_chall_readability_score=0
        list_features["dale_chall_readability_score"]=dale_chall_readability_score
        end=time.time()
        ex_time=end-start
        list_time["dale_chall_readability_score"]=ex_time
        #list_features[""]=

def Email_Body_difficult_words(body, list_features, list_time):
    if config["Email_Body_Features"]["difficult_words"] == "True":
        start=time.time()
        if body:
            try:
                difficult_words=textstat.difficult_words(body)
            except Exception as e:
                logger.warning("exception: " + str(e))
                difficult_words=-1
        else:
            difficult_words=0
        list_features["difficult_words"]=difficult_words
        end=time.time()
        ex_time=end-start
        list_time["difficult_words"]=ex_time

def Email_Body_linsear_score(body, list_features, list_time):
    if config["Email_Body_Features"]["linsear_score"] == "True":
        start=time.time()
        if body:
            try:
                linsear_score=textstat.linsear_write_formula(body)
            except Exception as e:
                logger.warning("exception: " + str(e))
                linsear_score=-1
        else:
            linsear_score=0
        list_features["linsear_score"]=linsear_score
        end=time.time()
        ex_time=end-start
        list_time["linsear_score"]=ex_time
        #list_features[""]=

def Email_Body_gunning_fog(body, list_features, list_time):
    if config["Email_Body_Features"]["gunning_fog"] == "True":
        start=time.time()
        if body:
            try:
                gunning_fog=textstat.gunning_fog(body)
            except Exception as e:
                logger.warning("exception: " + str(e))
                gunning_fog=-1
        else:
            gunning_fog=0
        list_features["gunning_fog"]=gunning_fog
        end=time.time()
        ex_time=end-start
        list_time["gunning_fog"]=ex_time

def Email_Body_text_standard(body, list_features, list_time):
    if config["Email_Body_Features"]["text_standard"] == "True":
        start=time.time()
        if body:
            try:
                text_standard=textstat.text_standard(body)
            except Exception as e:
                logger.warning("exception: " + str(e))
                text_standard=-1
        else:
            text_standard=0
        list_features["text_standard"]=text_standard
        end=time.time()
        ex_time=end-start
        list_time["text_standard"]=ex_time

#### Extract features from subject
def Email_Header_number_of_words_subject(subject, list_features, list_time):
    if config["Email_Header_Features"]["number_of_words_subject"] == "True":
        start=time.time()
        if subject:
            try:
                logger.debug("subject: {}".format(subject))
                number_of_words_subject = len(re.findall(r'\w+', subject))
            except Exception as e:
                number_of_words_subject=-1
                logger.warning("exception: " + str(e))
        else:
            number_of_words_subject=0
        list_features["number_of_words_subject"]=number_of_words_subject
        end=time.time()
        ex_time=end-start
        list_time["number_of_words_subject"]=ex_time

def Email_Header_number_of_characters_subject(subject, list_features, list_time):
    if config["Email_Header_Features"]["number_of_characters_subject"] == "True":
        start=time.time()
        if subject:    
            try:
                number_of_characters_subject = len(re.findall(r'\w', subject))
            except Exception as e:
                number_of_characters_subject=-1
                logger.warning("exception: " + str(e))
        else:
            number_of_characters_subject = 0
        list_features["number_of_characters_subject"]=number_of_characters_subject
        end=time.time()
        ex_time=end-start
        list_time["number_of_characters_subject"]=ex_time

def Email_Header_number_of_special_characters_subject(subject, list_features, list_time):
    if config["Email_Header_Features"]["number_of_special_characters_subject"] == "True":
        start=time.time()
        if subject:    
            try:
                number_of_characters_subject = len(re.findall(r'\w', subject))
                number_of_special_characters_subject = len(subject)-number_of_characters_subject-len(re.findall(r' ', subject))
            except Exception as e:
                number_of_special_characters_subject=-1
                logger.warning("exception: " + str(e))
        else:
            number_of_special_characters_subject=0
        list_features["number_of_special_characters_subject"]=number_of_special_characters_subject
        end=time.time()
        ex_time=end-start
        list_time["number_of_special_characters_subject"]=ex_time
    

def Email_Header_binary_fwd(subject, list_features, list_time):
    if config["Email_Header_Features"]["binary_fwd"] == "True":
        start=time.time()
        if subject:    
            try:
                email_forward_subject = re.compile(r"^FW:", flags=re.IGNORECASE)
                binary_fwd= int(bool(re.search(email_forward_subject, subject)))
            except Exception as e:
                binary_fwd=-1
                logger.warning("exception: " + str(e))
        else:
            binary_fwd = 0
        list_features["binary_fwd"]=binary_fwd
        end=time.time()
        ex_time=end-start
        list_time["binary_fwd"]=ex_time

def Email_Header_binary_re(subject, list_features, list_time):
    if config["Email_Header_Features"]["binary_re"] == "True":
        start=time.time()
        if subject:    
            try:
                email_re_subject = re.compile(r"^re:", flags=re.IGNORECASE)
                binary_re= int(bool(re.search(email_re_subject, subject)))
            except Exception as e:
                binary_re=-1
                logger.warning("exception: " + str(e))
        else:
            binary_re=0
        list_features["binary_fwd"]=binary_fwd
        end=time.time()
        ex_time=end-start
        list_time["binary_fwd"]=ex_time

def Email_Header_vocab_richness_subject(subject, list_features, list_time):
    if config["Email_Header_Features"]["vocab_richness_subject"] == "True":
        start=time.time()
        if subject:
            try:
                vocab_richness_subject=yule(subject)
            except Exception as e:
                vocab_richness_subject=-1
                logger.warning("exception: " + str(e))
        else:
            vocab_richness_subject=0
        list_features["vocab_richness_subject"]=vocab_richness_subject
        end=time.time()
        ex_time=end-start
        list_time["vocab_richness_subject"]=ex_time



############################ HTML features
def HTML_number_of_tags(soup, list_features, list_time):
    if config["HTML_Features"]["number_of_tags"] == "True":
        start=time.time()
        number_of_tags=0
        if soup:
            try:
                all_tags = soup.find_all()
                number_of_tags = len(all_tags)
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_tags=-1
        else:
            number_of_tags=0
        list_features["number_of_tags"]=number_of_tags
        end=time.time()
        ex_time=end-start
        list_time["number_of_tags"]=ex_time

def HTML_number_of_head(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_head"] == "True":
        start=time.time()
        number_of_head=0
        if soup:
            try:
                number_of_head = len(soup.find_all('head'))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_head=-1
        list_features["number_of_head"]=number_of_head
        end=time.time()
        ex_time=end-start
        list_time["number_of_head"]=ex_time

def HTML_number_of_html(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_html"] == "True":
        start=time.time()
        number_of_html=0
        if soup:    
            try:
                number_of_html = len(soup.find_all('html'))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_html=-1
        list_features["number_of_html"]=number_of_html
        end=time.time()
        ex_time=end-start
        list_time["number_of_html"]=ex_time

def HTML_number_of_body(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_body"] == "True":
        start=time.time()
        number_of_body=0
        if soup:    
            try:
                number_of_body = len(soup.find_all('body'))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_body = -1
        list_features["number_of_body"]=number_of_body
        end=time.time()
        ex_time=end-start
        list_time["number_of_body"]=ex_time

def HTML_number_of_titles(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_titles"] == "True":
        start=time.time()
        number_of_titles=0
        if soup:    
            try:
                number_of_titles = len(soup.find_all('title'))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_titles=-1
        list_features["number_of_titles"]=number_of_titles
        end=time.time()
        ex_time=end-start
        list_time["number_of_titles"]=ex_time

def HTML_number_suspicious_content(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_suspicious_content"] == "True":
        start=time.time()
        all_tags = soup.find_all()
        number_suspicious_content = 0
        if soup:
            try:
                for tag in all_tags:
                    str_tag = str(tag)
                    if  len(str_tag) > 128 and (str_tag.count(' ')/len(str_tag) < 0.05):
                        number_suspicious_content = number_suspicious_content + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_suspicious_content=-1
        list_features["number_suspicious_content"]=number_suspicious_content
        end=time.time()
        ex_time=end-start
        list_time["number_suspicious_content"]=ex_time

def HTML_number_of_iframes(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_iframes"] == "True":
        start=time.time()
        number_of_iframes=0
        if soup:    
            try:
                iframe_tags = soup.find_all('iframe')
                number_of_iframes = len(iframe_tags)
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_iframes=-1
        list_features["number_of_iframes"]=number_of_iframes
        end=time.time()
        ex_time=end-start
        list_time["number_of_iframes"]=ex_time

def HTML_number_of_input(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_input"] == "True":
        start=time.time()
        number_of_input=0
        if soup:
            try:
                number_of_input = len(soup.find_all('input'))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_input = -1
        list_features["number_of_input"]=number_of_input
        end=time.time()
        ex_time=end-start
        list_time["number_of_input"]=ex_time

def HTML_number_of_img(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_img"] == "True":
        start=time.time()
        number_of_img=0
        if soup:
            try:
                number_of_img = len(soup.find_all('img'))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_img = -1
        list_features["number_of_img"]=number_of_img
        end=time.time()
        ex_time=end-start
        list_time["number_of_img"]=ex_time


def HTML_number_of_scripts(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_scripts"] == "True":
        start=time.time()
        number_of_scripts=0
        if soup:
            try:
                number_of_scripts = len(soup.find_all('script'))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_scripts = -1
        list_features["number_of_scripts"]=number_of_scripts
        end=time.time()
        ex_time=end-start
        list_time["number_of_scripts"]=ex_time

def HTML_number_of_anchor(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_anchor"] == "True":
        start=time.time()
        number_of_anchor=0
        if soup:
            try:
                number_of_anchor = len(soup.find_all('a'))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_anchor=-1
        list_features["number_of_anchor"]=number_of_anchor
        end=time.time()
        ex_time=end-start
        list_time["number_of_anchor"]=ex_time

def HTML_number_of_embed(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_embed"] == "True":
        start=time.time()
        number_of_embed=0
        if soup:
            try:
                number_of_embed = len(soup.find_all('embed'))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_embed=-1
        list_features["number_of_embed"]=number_of_embed
        end=time.time()
        ex_time=end-start
        list_time["number_of_embed"]=ex_time

def HTML_number_object_tags(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_object_tags"] == "True":
        start=time.time()
        number_object_tags=0
        if soup:
            try:
                object_tags = len(soup.find_all('object'))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_object_tags = -1
        list_features["number_object_tags"]=number_object_tags
        end=time.time()
        ex_time=end-start
        list_time["number_object_tags"]=ex_time

def HTML_number_of_video(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_video"] == "True":
        start=time.time()
        number_of_video=0
        if soup:
            try:
                number_of_video = len(soup.find_all('video'))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_video = -1
        list_features["number_of_video"]=number_of_video
        end=time.time()
        ex_time=end-start
        list_time["number_of_video"]=ex_time

def HTML_number_of_audio(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_audio"] == "True":
        start=time.time()
        number_of_audio=0
        if soup:
            try:
                number_of_audio = len(soup.find_all('audio'))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_audio=-1
        list_features["number_of_audio"]=number_of_audio
        end=time.time()
        ex_time=end-start
        list_time["number_of_audio"]=ex_time

def HTML_number_of_hidden_iframe(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_hidden_iframe"] == "True":
        start=time.time()
        number_of_hidden_iframe = 0
        if soup:   
            try:
                iframe_tags = soup.find_all('iframe') 
                for tag in iframe_tags:
                    if tag.get('height') == 0 or tag.get('width') == 0:
                        number_of_hidden_iframe = number_of_hidden_iframe + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_hidden_iframe= -1
        list_features["number_of_hidden_iframe"]=number_of_hidden_iframe
        end=time.time()
        ex_time=end-start
        list_time["number_of_hidden_iframe"]=ex_time

def HTML_number_of_hidden_div(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_hidden_div"] == "True":
        start=time.time()
        number_of_hidden_div=0
        if soup:
            try:
                tags = soup.find_all('div')
                for tag in tags:
                    if tag.get('height') == 0 or tag.get('width') == 0:
                        number_of_hidden_div = number_of_hidden_div + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_hidden_div=-1
        list_features["number_of_hidden_div"]=number_of_hidden_div
        end=time.time()
        ex_time=end-start
        list_time["number_of_hidden_div"]=ex_time

def HTML_number_of_hidden_object(soup, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["number_of_hidden_object"] == "True":
        start=time.time()
        number_of_hidden_object = 0
        if soup:    
            try:
                object_tags = soup.find_all('object')
                for tag in object_tags:
                    if tag.get('height') == 0 or tag.get('width') == 0:
                        number_of_hidden_object = number_of_hidden_object + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_hidden_object=-1
        list_features["number_of_hidden_object"]=number_of_hidden_object
        end=time.time()
        ex_time=end-start
        list_time["number_of_hidden_object"]=ex_time

def HTML_inbound_count(soup, url, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["inbound_count"] == "True":
        start=time.time()
        inbound_count = 0
        if soup:
            try:
                url_extracted = tldextract.extract(url)
                local_domain = '{}.{}'.format(url_extracted .domain, url_extracted .suffix)
                tags = soup.find_all(['audio', 'embed', 'iframe', 'img', 'input', 'script', 'source', 'track', 'video'])
                for tag in tags:
                    src_address = tag.get('src')
                    if src_address != None:
                        if src_address.lower().startswith(("https", "http")):
                            extracted = tldextract.extract(src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link == local_domain:
                                inbound_count = inbound_count + 1
                        elif src_address.startswith("//"):
                            extracted = tldextract.extract("http:" + src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link == local_domain:
                                inbound_count = inbound_count + 1
                        else:
                            inbound_count = inbound_count + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                inbound_count=-1
        list_features["inbound_count"]=inbound_count
        end=time.time()
        ex_time=end-start
        list_time["inbound_count"]=ex_time

def HTML_outbound_count(soup, url, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["outbound_count"] == "True":
        start=time.time()
        outbound_count = 0
        if soup:
            try:
                url_extracted = tldextract.extract(url)
                local_domain = '{}.{}'.format(url_extracted .domain, url_extracted .suffix)
                tags = soup.find_all(['audio', 'embed', 'iframe', 'img', 'input', 'script', 'source', 'track', 'video'])
                for tag in tags:
                    src_address = tag.get('src')
                    if src_address != None:
                        if src_address.lower().startswith(("https", "http")):
                            extracted = tldextract.extract(src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link != local_domain:
                                outbound_count = outbound_count + 1
                        elif src_address.startswith("//"):
                            extracted = tldextract.extract("http:" + src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link != local_domain:
                                outbound_count = outbound_count + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                outbound_count=-1
        list_features["outbound_count"]=outbound_count
        end=time.time()
        ex_time=end-start
        list_time["outbound_count"]=ex_time

def HTML_inbound_href_count(soup, url, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["inbound_href_count"] == "True":
        start=time.time()
        inbound_href_count = 0
        if soup:
            try:    
                url_extracted = tldextract.extract(url)
                local_domain = '{}.{}'.format(url_extracted.domain, url_extracted.suffix)
                tags = soup.find_all(['a', 'area', 'base', 'link'])
                for tag in tags:
                    src_address = tag.get('href')
                    if src_address is not None:
                        if src_address.lower().startswith(("https", "http")):
                            extracted = tldextract.extract(src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link == local_domain:
                                inbound_href_count = inbound_href_count + 1
                        elif src_address.startswith("//"):
                            extracted = tldextract.extract("http:" + src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link == local_domain:
                                inbound_href_count = inbound_href_count + 1
                        else:
                            inbound_href_count = inbound_href_count + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                inbound_href_count=-1
        list_features["inbound_href_count"]=inbound_href_count
        end=time.time()
        ex_time=end-start
        list_time["inbound_href_count"]=ex_time

def HTML_outbound_href_count(soup, url, list_features, list_time):
    #global list_features
    if config["HTML_Features"]["outbound_href_count"] == "True":
        start=time.time()
        outbound_href_count = 0
        if soup:    
            try:
                url_extracted = tldextract.extract(url)
                local_domain = '{}.{}'.format(url_extracted.domain, url_extracted.suffix)
                tags = soup.find_all(['a', 'area', 'base', 'link'])
                for tag in tags:
                    src_address = tag.get('href')
                    if src_address is not None:
                        if src_address.lower().startswith(("https", "http")):
                            extracted = tldextract.extract(src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link != local_domain:
                                outbound_href_count = outbound_href_count + 1
                        elif src_address.startswith("//"):
                            extracted = tldextract.extract("http:" + src_address)
                            filtered_link = '{}.{}'.format(extracted.domain, extracted.suffix)
                            if filtered_link != local_domain:
                                outbound_href_count = outbound_href_count + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                outbound_href_count=-1
        list_features["outbound_href_count"]=outbound_href_count
        end=time.time()
        ex_time=end-start
        list_time["outbound_href_count"]=ex_time

def HTML_Website_content_type(html, list_features, list_time):
    if config["HTML_Features"]["website_content_type"] == "True":
        start=time.time()
        #print(html.headers)
        if html:
            try:
                if 'Content-Type' in html.headers:
                    content_type = html.headers['Content-Type'].split(';')[0]
                else:
                    content_type = 'text/html'
            except Exception as e:
                logger.warning("exception: " + str(e))
                content_type="N/A"
        else:
            content_type=''
        list_features["Website_content_type"]=content_type    
        end=time.time()
        ex_time=end-start
        list_time["content_type"]=ex_time

def HTML_content_length(html, list_features, list_time):
    if config["HTML_Features"]["content_length"] == "True":
        start=time.time()
        content_length = 0
        if html:
            try:
                if 'Content-Length' in html.headers:
                    content_length = html.headers['Content-Length']
            except Exception as e:
                logger.warning("exception: " + str(e))
                content_length=-1
        list_features["content_length"]=int(content_length)
        end=time.time()
        ex_time=end-start
        list_time["content_length"]=ex_time

def HTML_x_powered_by(html, list_features, list_time):
    if config["HTML_Features"]["x_powered_by"] == "True":
        start=time.time()
        x_powered_by = ''
        if html:
            try:
                if 'X-Powered-By' in html.headers:
                    #x_powered_by = html.headers['X-Powered-By']
                    x_powered_by = html.headers["X-Powered-By"]
            except Exception as e:
                logger.warning("exception: " + str(e))
                x_powered_by = "N/A"
        list_features["x_powered_by"]=x_powered_by
        end=time.time()
        ex_time=end-start
        list_time["x_powered_by"]=ex_time

def HTML_URL_Is_Redirect(html, url, list_features, list_time):
    if config["HTML_Features"]["URL_Is_Redirect"]=="True":
        start=time.time()
        flag=0
        if html:
            try:
                if url != html.url:
                    flag=1
            except Exception as e:
                logger.warning("Exception: {}".format(e))
                flag=-1
        list_features["URL_Is_Redirect"]=flag
        end=time.time()
        ex_time=end-start
        list_time["URL_Is_Redirect"]=ex_time

def HTML_Is_Login(html, url, list_features, list_time):
    if config["HTML_Features"]["Is_Login"]=="True":
        start=time.time()
        userfield = passfield = emailfield = None
        _is_login = False
        doc = lxml_html.document_fromstring(html, base_url=url)
        try:
            form_element = doc.xpath('//form')
            if form_element:
                form = _pick_form(form_element)
            else:
                return _is_login
            for x in form.inputs:
                if not isinstance(x, html.InputElement):
                    continue
                type_ = x.type
                if type_ == 'password' and passfield is None:
                    passfield = x.name
                    _is_login = True
                    break
        except Exception as ex:
            _is_login = False

        list_features['is_login'] = _is_login
        end = time.time()
        ex_time=end-start
        list_time['is_login'] = ex_time

############################ URL features
def URL_url_length(url, list_features, list_time):
    ##global list_features
    if config["URL_Features"]["url_length"] == "True":
        start=time.time()
        url_length=0
        if url:
            try:
                url_length=len(url)
            except Exception as e:
                logger.warning("exception: " + str(e))
                url_length=-1
        list_features["url_length"]=url_length
        end=time.time()
        ex_time=end-start
        list_time["url_length"]=ex_time


def URL_domain_length(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["domain_length"] == "True":
        start=time.time()
        domain_length=0
        if url:    
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.hostname
                domain_length = len(domain)
            except Exception as e:
                logger.warning("exception: " + str(e))
                domain_length=-1
        list_features["domain_length"]=domain_length
        end=time.time()
        ex_time=end-start
        list_time["domain_length"]=ex_time

##################################################################################
def URL_letter_occurence(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["letter_occurence"] == "True":
        start=time.time()
        if url:
            ####
            try:
                domain = '{uri.scheme}://{uri.hostname}/'.format(uri=parsed_url).lower()
            except Exception as e:
                logger.warning("exception: " + str(e))
                for x in range(26):
                    list_features["letter_occurence_"+chr(x+ ord('a'))]=-1
            ####   
            for x in range(26):
                try:
                    list_features["letter_occurence_"+chr(x+ ord('a'))]=domain.count(chr(x + ord('a')))
                except Exception as e:
                    logger.warning("exception: " + str(e))
                    list_features["letter_occurence_"+chr(x+ ord('a'))]=-1
        else:
            for x in range(26):
                list_features["letter_occurence_"+chr(x+ ord('a'))]=0
        end=time.time()
        ex_time=end-start
        list_time["letter_occurence"]=ex_time
        #print("letter_occurence >>>>>>>>>>>>>>>>>>: " + str(letter_occurence))
        #list_features["letter_occurence"]=letter_occurence

##################################################################################
def URL_char_distance(url, list_features, list_time):
    if config["URL_Features"]["char_distance"] == "True":
        start=time.time()
        if url:
            count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1))) 
            for x in range(26):
                try:
                    url_char_dist=(url.count(chr(x + ord('a'))) / (count(url,string.ascii_letters)))
                    list_features["url_char_distance_"+chr(x + ord('a'))]=url_char_dist
                except Exception as e:
                    logger.warning("exception: " + str(e))
                    list_features["url_char_distance_"+chr(x + ord('a'))]=-1
        else:
            for x in range(26):
                list_features["url_char_distance_"+chr(x + ord('a'))]=0
        end=time.time()
        ex_time=end-start
        list_time["url_char_distance_"]=ex_time

##################################################################################
def URL_kolmogorov_shmirnov(list_features, list_time):
    if config["URL_Features"]["kolmogorov_shmirnov"] == "True":
        start=time.time()
        char_dist = [.08167, .01492, .02782, .04253, .12702, .02228, .02015, .06094, .06966, .00153, .00772, .04025, .02406,
                 .06749, .07507, .01929, .00095, .05987, .06327, .09056, .02758, .00978, .02360, .00150, .01974, .00074]
        
        try:
            #if list_features.get("url_char_distance") == 0:
            #    list_features["kolmogorov_shmirnov"]= 0
            #else:
            url_char_distance=[]
            for x in range(26):
                url_char_distance.append(list_features["url_char_distance_" + chr(x+ ord('a'))])
            if any(distance == -1 for distance in url_char_distance):
                ks=-1
            else:
                ks = stats.ks_2samp(url_char_distance, char_dist)
        except Exception as e:
            logger.warning("exception: " + str(e))
            ks=-1
        if ks==-1:
            list_features["kolmogorov_shmirnov"]=ks
        else:
            list_features["kolmogorov_shmirnov"]=ks[0]
        end=time.time()
        ex_time=end-start
        list_time["kolmogorov_shmirnov"]=ex_time

def URL_Kullback_Leibler_Divergence(list_features, list_time):
    if config["URL_Features"]["Kullback_Leibler_Divergence"] == "True":
        start=time.time()
        char_dist = [.08167, .01492, .02782, .04253, .12702, .02228, .02015, .06094, .06966, .00153, .00772, .04025, .02406,
                 .06749, .07507, .01929, .00095, .05987, .06327, .09056, .02758, .00978, .02360, .00150, .01974, .00074]
        try:
            url_char_distance=[]
            for x in range(26):
                url_char_distance.append(list_features["url_char_distance_" + chr(x+ ord('a'))])
            if any(distance == -1 for distance in url_char_distance):
                kl=-1
            else:
                kl = stats.entropy(url_char_distance, char_dist)
        except Exception as e:
            logger.warning("exception: " + str(e))
            kl=-1
        logger.debug("KL: >>>> {}".format(kl))
        list_features["Kullback_Leibler_Divergence"]=kl
        end=time.time()
        ex_time=end-start
        list_time["Kullback_Leibler_Divergence"]=ex_time

def URL_english_frequency_distance(list_features, list_time):
    #global list_features
    if config["URL_Features"]["english_frequency_distance"] == "True":
        start=time.time()
        char_dist = [.08167, .01492, .02782, .04253, .12702, .02228, .02015, .06094, .06966, .00153, .00772, .04025, .02406,
                 .06749, .07507, .01929, .00095, .05987, .06327, .09056, .02758, .00978, .02360, .00150, .01974, .00074]
        try:
            #if list_features.get("url_char_distance") is None:
            #    list_features["edit_distance"]= 0
            #else:
            url_char_distance=[]
            for x in range(26):
                url_char_distance.append(list_features["url_char_distance_" + chr(x+ ord('a'))])
            if any(distance==-1 for distance in url_char_distance):
                ed=-1
            else:
                ed = distance.euclidean(url_char_distance, char_dist)
        except Exception as e:
            logger.warning("exception: " + str(e))
            ed=-1
        list_features["edit_distance"]=ed
        end=time.time()
        ex_time=end-start
        list_time["edit_distance"]=ex_time

def URL_num_punctuation(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["num_punctuation"] == "True":
        start=time.time()
        num_punct=0
        if url:
            try:
                count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
                num_punct = count(url, string.punctuation)
            except Exception as e:
                logger.warning("exception: " + str(e))
                num_punct=-1
        list_features["num_punctuation"]=num_punct
        end=time.time()
        ex_time=end-start
        list_time["num_punctuation"]=ex_time

def URL_has_port(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["has_port"] == "True":
        start=time.time()
        has_port=0
        if url:
            try:
                parsed_url=urlparse(url)
                port_number = '{uri.port}'.format(uri=parsed_url)
                has_port = 1
                if port_number == 'None':
                    has_port = 0
            except Exception as e:
                logger.warning("exception: " + str(e))
                has_port=-1
        list_features["has_port"]=has_port
        end=time.time()
        ex_time=end-start
        list_time["has_port"]=ex_time

def URL_has_https(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["has_https"] == "True":
        start=time.time()
        has_https=0
        if url:    
            try:
                parsed_url=urlparse(url)
                domain = '{uri.scheme}://{uri.hostname}/'.format(uri=parsed_url)
                has_https = 0
                if domain.startswith("https:"):
                    has_https = 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                has_https = -1
        list_features["has_https"]=has_https
        end=time.time()
        ex_time=end-start
        list_time["has_https"]=ex_time

def URL_number_of_digits(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["number_of_digits"] == "True":
        number_of_digits=0
        start=time.time()
        if url:
            try:
                number_of_digits = sum(c.isdigit() for c in url)
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_digits=-1
        list_features["number_of_digits"]=number_of_digits
        end=time.time()
        ex_time=end-start
        list_time["number_of_digits"]=ex_time

def URL_number_of_dots(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["number_of_dots"] == "True":
        start=time.time()
        number_of_dots=0
        if url:
            try:
                number_of_dots=url.count('.')
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_dots=-1
        list_features["number_of_dots"]=number_of_dots
        end=time.time()
        ex_time=end-start
        list_time["number_of_dots"]=ex_time

def URL_number_of_slashes(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["number_of_slashes"] == "True":
        start=time.time()
        number_of_slashes=0
        if url:
            try:
                number_of_slashes = url.count('/')
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_slashes=-1
        list_features["number_of_slashes"]=number_of_slashes
        end=time.time()
        ex_time=end-start
        list_time["number_of_slashes"]=ex_time

def URL_digit_letter_ratio(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["digit_letter_ratio"] == "True":
        start=time.time()
        digit_letter_ratio=0
        if url:
            try:
                number_of_digits = sum(c.isdigit() for c in url)
                letters = sum(c.isalpha() for c in url)
                digit_letter_ratio = number_of_digits/letters
                list_features["digit_letter_ratio"]=digit_letter_ratio
            except Exception as e:
                logger.warning("exception: " + str(e))
                list_features["digit_letter_ratio"]=-1
        list_features["digit_letter_ratio"]= digit_letter_ratio
        end=time.time()
        ex_time=end-start
        list_time["digit_letter_ratio"]=ex_time

def URL_special_char_count(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["special_char_count"] == "True":
        start=time.time()
        special_char_count=0
        if url:
            try:
                special_char_count = url.count('@') + url.count('-')
            except Exception as e:
                logger.warning("exception: " + str(e))
                special_char_count=-1
        list_features["special_char_count"]=special_char_count
        end=time.time()
        ex_time=end-start
        list_time["special_char_count"]=ex_time

def URL_Top_level_domain(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["Top_level_domain"] == "True":
        start=time.time()
        tld=0
        if url:
            try:
                extracted = tldextract.extract(url)
                tld = "{}".format(extracted.suffix)
            except Exception as e:
                logger.warning("exception: " + str(e))
                tld=-1
        list_features["Top_level_domain"]=tld
        end=time.time()
        ex_time=end-start
        list_time["Top_level_domain"]=ex_time

def URL_Is_IP_Addr(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["Is_IP_Addr"] == "True":
        start=time.time()
        Is_IP_Addr=1
        if url:    
            try:
                parsed_url = urlparse(url)
                domain = '{uri.hostname}'.format(uri=parsed_url)
                if re.match("^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) == None:
                    Is_IP_Addr= 0
            except Exception as e:
                logger.warning("exception: " + str(e))
                Is_IP_Addr=-1

        list_features["Is_IP_Addr"]=Is_IP_Addr
        end=time.time()
        ex_time=end-start
        list_time["Is_IP_Addr"]=ex_time

# Devin's features
def URL_number_of_dashes(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["number_of_dashes"] == "True":
        start=time.time()
        number_of_dashes=0
        if url:    
            try:
                number_of_dashes = url.count('-')
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_dashes = -1
        list_features["number_of_dashes"]=number_of_dashes
        end=time.time()
        ex_time=end-start
        list_time["number_of_dashes"]=ex_time

def URL_Http_middle_of_URL(url, list_features, list_time):
    #global list_features
    if config["URL_Features"]["Http_middle_of_URL"] == "True":
        start=time.time()
        Http_middle_of_URL=0
        #regex_http=re.compile(r'')
        if url:    
            try:
                if 'http' in url and url.startswith('http') == False:
                    Http_middle_of_URL=1
            except Exception as e:
                logger.warning("exception: " + str(e))
                Http_middle_of_URL=-1
        list_features["Http_middle_of_URL"]=Http_middle_of_URL
        end=time.time()
        ex_time=end-start
        list_time["Http_middle_of_URL"]=ex_time

def URL_Has_More_than_3_dots(url, list_features, list_time):
    if config["URL_Features"]["Has_More_than_3_dots"] == "True":
        start=time.time()
        #regex_http=re.compile(r'')
        if url:
            try:
                url=url.replace('www.','')
                count_dots=url.count('.')
                if count_dots >= 3:
                    list_features["Has_More_than_3_dots"]=1
                else:
                    list_features["Has_More_than_3_dots"]=0
            except Exception as e:
                logger.warning("exception: " + str(e))
                list_features["Has_More_than_3_dots"]=-1
        else:
            list_features["Has_More_than_3_dots"]=0
        end=time.time()
        ex_time=end-start
        list_time["Has_More_than_3_dots"]=ex_time

def URL_Has_at_symbole(url, list_features, list_time):
    if config["URL_Features"]["Has_at_symbole"] == "True":
        start=time.time()
        flag=0
        if url:
            try:
                if "@" in url:
                    flag=1
            except Exception  as e:
                logger.warning("Exception: " + str(e))
                flag=-1
        list_features["Has_at_symbole"]=flag
        end=time.time()
        ex_time=end-start
        list_time["Has_at_symbole"]=ex_time


def URL_Has_anchor_tag(url, list_features, list_time):
    if config["URL_Features"]["Has_anchor_tag"] == "True":
        start=time.time()
        regex_anchor=re.compile(r'<\?a>')
        flag=0
        if url:
            try:
                flag=int(bool(re.findall(regex_anchor,url)))
            except Exception  as e:
                logger.warning("Exception: " + str(e))
                flag=-1
        list_features["Has_anchor_tag"]=flag
        end=time.time()
        ex_time=end-start
        list_time["Has_anchor_tag"]=ex_time

def URL_Null_in_Domain(url, list_features, list_time):
    if config["URL_Features"]["Null_in_Domain"] == "True":
        start=time.time()
        regex_null=re.compile(r'null', flags=re.IGNORECASE)
        flag=0
        if url:
            try:
                flag=int(bool(re.findall(regex_null,url)))
            except Exception  as e:
                logger.warning("Exception: " + str(e))
                flag=-1
        list_features["Null_in_Domain"]=flag
        end=time.time()
        ex_time=end-start
        list_time["Null_in_Domain"]=ex_time

def URL_Token_Count(url, list_features, list_time):
    if config["URL_Features"]["Token_Count"] == "True":
        start=time.time()
        count=0
        if url:
            try:
                count=len(url.split(config["URL_Features"]["URL_token_delimiter"]))
            except Exception  as e:
                logger.warning("Exception: " + str(e))
                count=-1
        list_features["Token_Count"]=count
        end=time.time()
        ex_time=end-start
        list_time["Token_Count"]=ex_time

def URL_Average_Path_Token_Length(url, list_features, list_time):
    if config["URL_Features"]["Average_Path_Token_Length"] == "True":
        start=time.time()
        average_token_length=0
        delimiters_regex=re.compile('[=|,|/|?|.|-]')
        if url:
            try:
                parsed_url=urlparse(url)
                path='{uri.path}'.format(uri=parsed_url)
                list_tokens=re.split(delimiters_regex,path)
                list_len_tokens=[0 for x in range(len(list_tokens))]
                for token in list_tokens:
                    list_len_tokens[list_tokens.index(token)]=len(token)
                average_token_length= sum(list_len_tokens)/len(list_len_tokens)
            except Exception  as e:
                logger.warning("Exception: " + str(e))
                average_token_length=-1
        list_features["Average_Path_Token_Length"]=average_token_length
        end=time.time()
        ex_time=end-start
        list_time["Average_Path_Token_Length"]=ex_time

def URL_Average_Domain_Token_Length(url, list_features, list_time):
    if config["URL_Features"]["Average_Domain_Token_Length"] == "True":
        start=time.time()
        average_token_length=0
        if url:
            try:
                parsed_url=urlparse(url)
                domain='{uri.hostname}'.format(uri=parsed_url)
                list_len_tokens=[]
                list_tokens=domain.split(config["URL_Features"]["URL_token_delimiter"])
                for token in list_tokens:
                    list_len_tokens.append(len(token))
                average_token_length= sum(list_len_tokens)/len(list_len_tokens)
            except Exception  as e:
                logger.warning("Exception: " + str(e))
                average_token_length=-1
        list_features["Average_Domain_Token_Length"]=average_token_length
        end=time.time()
        ex_time=end-start
        list_time["Average_Domain_Token_Length"]=ex_time

def URL_Longest_Domain_Token(url, list_features, list_time):
    if config["URL_Features"]["Longest_Domain_Token"] == "True":
        start=time.time()
        longest_token_len=0
        try:
            if url=='':
                longest_token_len=0
            else:
                parsed_url=urlparse(url)
                domain='{uri.hostname}'.format(uri=parsed_url)
                list_len_tokens=[]
                list_tokens=domain.split(config["URL_Features"]["URL_token_delimiter"])
                for token in list_tokens:
                    list_len_tokens.append(len(token))
                longest_token_len=max(list_len_tokens)
                
        except Exception  as e:
            logger.warning("Exception: " + str(e))
            longest_token_len=-1
        list_features["Longest_Domain_Token"]=longest_token_len
        end=time.time()
        ex_time=end-start
        list_time["Longest_Domain_Token"]=ex_time

def URL_Protocol_Port_Match(url, list_features, list_time):
    if config["URL_Features"]["Protocol_Port_Match"]=="True":
        start=time.time()
        match = 1
        if url:
            try:
                parsed_url = urlparse(url)
                scheme = '{uri.scheme}'.format(uri=parsed_url).lower()
                port = '{uri.port}'.format(uri=parsed_url)
                protocol_port_list=[('http',8080), ('http',80), ('https',443), ('ftp',20), ('tcp',20), ('scp',20),('ftp',21), ('ssh',22), ('telnet',23), ('smtp',25), ('dns',53), ("pop3", 110), ("sftp", 115), ("imap", 143), ("smtp",465), ("rlogin", 513), ("imap", 993), ("pop3", 995)]
                if port != 'None' and ((scheme, int(port)) not in protocol_port_list):
                    match = 0
                list_features["Protocol_Port_Match"] = match
            except Exception as e:
                logger.warning("Exception: {}".format(e))
                match=-1
        else:
            match=0
        list_features["Protocol_Port_Match"]=match
        end=time.time()
        ex_time=end-start
        list_time["Protocol_Port_Match"]=ex_time

def URL_Has_WWW_in_Middle(url, list_features, list_time):
    if config["URL_Features"]["Has_WWW_in_Middle"] == "True":
        start=time.time()
        flag=0
        #regex_www=re.compile(r'www')
        if url:
            try:
                parsed_url = urlparse(url)
                domain = '{uri.hostname}'.format(uri=parsed_url).lower()
                if 'www' in domain and domain.startswith('www') == False:
                    flag=1
            except Exception as e:
                logger.warning("exception: " + str(e))
                flag=-1
        list_features["Has_WWW_in_Middle"]=flag
        end=time.time()
        ex_time=end-start
        list_time["Has_WWW_in_Middle"]=ex_time

def URL_Has_Hex_Characters(url, list_features, list_time):
    if config['URL_Features']['Has_Hex_Characters']=="True":
        start=time.time()
        flag=0
        regex_hex=re.compile(r'%[1-9A-Z][1-9A-Z]')
        if url:
            try:
                #parsed_url = urlparse(url)
                #domain = '{uri.netloc}'.format(uri=parsed_url).lower()
                flag=int((bool(re.findall(regex_hex,url))))
            except Exception as e:
                logger.warning("Exception: {}".format(e))
                flag=-1
        list_features["Has_Hex_Characters"]=flag
        end=time.time()
        ex_time=end-start
        list_time["Has_Hex_Characters"]=ex_time

def URL_Double_Slashes_Not_Beginning_Count(url, list_features, list_time):
    if config['URL_Features']['Double_Slashes_Not_Beginning_Count']=="True":
        start=time.time()
        flag=0
        regex_2slashes=re.compile(r'//')
        if url:
            try:
                parsed_url = urlparse(url)
                path = '{uri.path}'.format(uri=parsed_url)
                flag=int((bool(re.findall(regex_2slashes,path))))
                list_features["Double_Slashes_Not_Beginning_Count"]=flag
            except Exception as e:
                logger.warning("Exception: {}".format(e))
                flag=-1
        list_features["Double_Slashes_Not_Beginning_Count"]=flag
        end=time.time()
        ex_time=end-start
        list_time["Double_Slashes_Not_Beginning_Count"]=ex_time

def URL_Brand_In_Url(url, list_features, list_time):
    if config['URL_Features']['Brand_In_Url']=="True":
        start=time.time()
        tokens = re.split('[^a-zA-Z]', url)
        brands = ['microsoft', 'paypal', 'netflix', 'bankofamerica', 'wellsfargo', 'facebook', 'chase', 'orange', 'dhl', 'dropbox', 'docusign', 'adobe', 'linkedin', 'apple', 'google', 'banquepopulaire', 'alibaba', 'comcast', 'credit', 'agricole', 'yahoo', 'at', 'nbc', 'usaa', 'americanexpress', 'cibc', 'amazon', 'ing', 'bt']
        if any(token.lower() in brands for token in tokens):
            list_features["Brand_In_URL"] = 1
        else:
            list_features["Brand_In_URL"] = 0

def URL_Is_Whitelisted(url, list_features, list_time):
    if config['URL_Features']['Is_Whitelisted']=="True":
        start=time.time()
        domain = tldextract.extract(url).domain
        whitelist = ['microsoft', 'paypal', 'netflix', 'bankofamerica', 'wellsfargo', 'facebook', 'chase', 'orange', 'dhl', 'dropbox', 'docusign', 'adobe', 'linkedin', 'apple', 'google', 'banquepopulaire', 'alibaba', 'comcast', 'credit', 'agricole', 'yahoo', 'at', 'nbc', 'usaa', 'americanexpress', 'cibc', 'amazon', 'ing', 'bt']
        if domain in whitelist:
            list_features["Is_Whitelisted"] = 1
        else:
            list_features["Is_Whitelisted"] = 0


#def URL_ foundURLProtocolAndPortDoNotMatch
############################ Network Features
#def registar_id(whois_info, registrar_mapping)
 #   registar_id = 0
  #  if 'registrar' in whois_info and whois_info['registrar'] in registrar_mapping:
   #     registar_id = registrar_mapping[whois_info['registrar']]
    #return registar_id


#def country(whois_info, list_features, list_time):
#    #global list_features
#    if config["Features"]["country"] == "True":
#        start=time.time()
#        country = "N/A"
#        try:
#            if 'country' in whois_info:
#                country = whois_info['country']
#        except Exception as e:
#            logger.warning("exception: " + str(e))
#        list_features["country"]=country
#        end=time.time()
#        ex_time=end-start
#        list_time["country"]=ex_time

# age of domain
def Network_creation_date(whois_info, list_features, list_time):
    #global list_features
    if config["Network_Features"]["creation_date"] == "True":
        start=time.time()
        creation_date = 0.0
        if whois_info:
            try:
                if "creation_date" in whois_info:
                    dateTime = whois_info.get("creation_date")
                    if dateTime is not None:
                        if type(dateTime) is list:
                            creation_date = dateTime[0].timestamp()
                        elif type(dateTime) is str:
                            creation_date = datetime(year = 1996, month = 1, day = 1).timestamp()
                        else:
                            creation_date = dateTime.timestamp()
            except Exception as e:
                logger.warning("exception: " + str(e))
                creation_date=-1
        list_features["creation_date"]=creation_date
        end=time.time()
        ex_time=end-start
        list_time["creation_date"]=ex_time

def Network_expiration_date(whois_info, list_features, list_time):
    #global list_features
    if config["Network_Features"]["expiration_date"] == "True":
        start=time.time()
        expiration_date=0.0
        if whois_info:
            try:
                if "expiration_date" in whois_info:
                    dateTime = whois_info.get("expiration_date")
                    if dateTime is not None:
                        if type(dateTime) is list:
                            expiration_date = dateTime[0].timestamp()
                        elif type(dateTime) is str:
                            expiration_date = 0.0
                        else:
                            expiration_date = dateTime.timestamp()
            except Exception as e:
                logger.warning("exception: " + str(e))
                expiration_date=-1
        list_features["expiration_date"]=expiration_date
        end=time.time()
        ex_time=end-start
        list_time["expiration_date"]=ex_time

def Network_updated_date(whois_info, list_features, list_time):
    if config["Network_Features"]["updated_date"] == "True":
        start=time.time()
        updated_date = 0.0
        if whois_info:    
            try:
                if "updated_date" in whois_info:
                    dateTime = whois_info.get("updated_date")
                    if dateTime is not None:
                        if type(dateTime) is list:
                            updated_date = dateTime[0].timestamp()
                        elif type(dateTime) is str:
                            updated_date = 0.0
                        else:
                            updated_date = dateTime.timestamp()
            except Exception as e:
                logger.warning("exception: " + str(e))
                updated_date=-1
        list_features["updated_date"]=updated_date
        #print("----Update_date: {}".format(updated_date))
        end=time.time()
        ex_time=end-start
        list_time["updated_date"]=ex_time

def Network_as_number(IP_whois_list, list_features, list_time):
    if config["Network_Features"]["as_number"] == "True":
        start=time.time()
        as_number = 0
        if IP_whois_list:    
            try: 
                if 'asn' in IP_whois_list:
                    as_number = IP_whois_list['asn']
            except Exception as e:
                logger.warning("exception: " + str(e))
                as_number=-1
        list_features["as_number"]=as_number
        end=time.time()
        ex_time=end-start
        list_time["as_number"]=ex_time

def Network_number_name_server(dns_info, list_features, list_time):
    if config["Network_Features"]["number_name_server"] == "True":
        start=time.time()
        number_name_server = 0
        if dns_info:    
            try:
                for val in dns_info: 
                    if 'NS' in val:
                        number_name_server += 1 
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_name_server=-1
        list_features["number_name_server"]=number_name_server
        end=time.time()
        ex_time=end-start
        list_time["number_name_server"]=ex_time

def Network_DNS_Info_Exists(url, list_features, list_time):
    if config["Network_Features"]["DNS_Info_Exists"]=="True":
        start=time.time()
        flag=1
        if url:
            try:
                parsed_url = urlparse(url)
                domain='{uri.hostname}'.format(uri=parsed_url)
                resolver = dns.resolver.Resolver()
                resolver.timeout = 3
                resolver.lifetime = 3
                try:
                    dns_info = resolver.query(domain, 'A')
                    flag=1
                except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers, dns.resolver.Timeout) as e:
                    logger.warning("Exception: {}".format(e))
                    flag=0
            except Exception as e:
                logger.warning("Exception: {}".format(e))
                flag=-1
                logger.debug(list_features["DNS_Info_Exists"])
        else:
            flag=0
        list_features["DNS_Info_Exists"]=flag
        end=time.time()
        ex_time=end-start
        list_time["DNS_Info_Exists"]=ex_time

def Network_dns_ttl(url, list_features, list_time):
    if config["Network_Features"]["dns_ttl"] == "True":
        start=time.time()
        dns_ttl = 0
        retry_count = 0
        if url:
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.hostname
            except Exception as e:
                logger.warning("exception: " + str(e))
                dns_ttl=-1
            try:
                while True:
                    try:
                        dns_complete_info = dns.resolver.query(domain, 'A')
                        dns_ttl = dns_complete_info.rrset.ttl
                    except dns.exception.Timeout:
                        if retry_count > 3: 
                            break
                        retry_count = retry_count + 1
                        continue
                    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers) as e:
                        logger.warning("Exception: {}".format(e))
                        dns_ttl=0
                        break
                    break
            except Exception as e:
                logger.warning("exception: " + str(e))
                dns_ttl=-1
        list_features["dns_ttl"]=dns_ttl
        end=time.time()
        ex_time=end-start
        list_time["dns_ttl"]=ex_time


############################ Javascript features
def Javascript_number_of_exec(soup, list_features, list_time):
    #global list_features
    if config["Javascript_Features"]["number_of_exec"] == "True":
        start=time.time()
        number_of_exec=0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'exec(' in script_text:
                            number_of_exec = number_of_exec + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_exec=-1
        list_features["number_of_exec"]=number_of_exec
        end=time.time()
        ex_time=end-start
        list_time["number_of_exec"]=ex_time

def Javascript_number_of_escape(soup, list_features, list_time):
    #global list_features
    if config["Javascript_Features"]["number_of_escape"] == "True":
        start=time.time()
        number_of_escape=0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'escape(' in script_text:
                            number_of_escape = number_of_escape + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_escape=-1
        list_features["number_of_escape"]=number_of_escape
        end=time.time()
        ex_time=end-start
        list_time["number_of_escape"]=ex_time

def Javascript_number_of_eval(soup, list_features, list_time):
    #global list_features
    if config["Javascript_Features"]["number_of_eval"] == "True":
        start=time.time()
        number_of_eval=0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'eval(' in script_text:
                            number_of_eval = number_of_eval + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_eval=-1
        list_features["number_of_eval"]=number_of_eval
        end=time.time()
        ex_time=end-start
        list_time["number_of_eval"]=ex_time

    
def Javascript_number_of_link(soup, list_features, list_time):
    #global list_features
    if config["Javascript_Features"]["number_of_link"] == "True":
        start=time.time()
        number_of_link=0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'link(' in script_text:
                            number_of_link = number_of_link + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_link=-1
        list_features["number_of_link"]=number_of_link
        end=time.time()
        ex_time=end-start
        list_time["number_of_link"]=ex_time

def Javascript_number_of_unescape(soup, list_features, list_time):
    #global list_features
    if config["Javascript_Features"]["number_of_unescape"] == "True":
        start=time.time()
        number_of_unescape=0
        scripts = soup.find_all('script')
        if soup:
            try:
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'unescape(' in script_text:
                            number_of_unescape = number_of_unescape + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_unescape=-1
        list_features["number_of_unescape"]=number_of_unescape
        end=time.time()
        ex_time=end-start
        list_time["number_of_unescape"]=ex_time

def Javascript_number_of_search(soup, list_features, list_time):
    #global list_features
    if config["Javascript_Features"]["number_of_search"] == "True":
        start=time.time()
        number_of_search=0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'search(' in script_text:
                            number_of_search = number_of_search + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_search=-1
        list_features["number_of_search"]=number_of_search
        end=time.time()
        ex_time=end-start
        list_time["number_of_search"]=ex_time

def Javascript_number_of_setTimeout(soup, list_features, list_time):
    #global list_features
    if config["Javascript_Features"]["number_of_setTimeout"] == "True":
        start=time.time()
        number_of_setTimeout=0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        if 'setTimeout(' in script_text:
                            number_of_setTimeout = number_of_setTimeout + 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_setTimeout=-1
        list_features["number_of_setTimeout"]=number_of_setTimeout
        end=time.time()
        ex_time=end-start
        list_time["number_of_setTimeout"]=ex_time

def Javascript_number_of_iframes_in_script(soup, list_features, list_time):
    #global list_features
    if config["Javascript_Features"]["number_of_iframes_in_script"] == "True":
        start=time.time()
        number_of_iframes_in_script=0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        number_of_iframes_in_script = number_of_iframes_in_script + script_text.count("iframe")
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_iframes_in_script=-1
        list_features["number_of_iframes_in_script"]=number_of_iframes_in_script
        end=time.time()
        ex_time=end-start
        list_time["number_of_iframes_in_script"]=ex_time

def Javascript_number_of_event_attachment(soup, list_features, list_time):
    #global list_features
    if config["Javascript_Features"]["number_of_event_attachment"] == "True":
        start=time.time()
        number_of_event_attachment=0
        if soup:
            try:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.get("type") is None or script.get("type") == 'text/javascript':
                        script_text = str(script)
                        number_of_event_attachment = number_of_event_attachment + len(re.findall(
                            "(?:addEventListener|attachEvent|dispatchEvent|fireEvent)\('(?:error|load|beforeunload|unload)'",
                            script_text.replace(" ", "")))
            except Exception as e:
                logger.warning("exception: " + str(e))
                number_of_event_attachment=-1
        list_features["number_of_event_attachment"]=number_of_event_attachment
        end=time.time()
        ex_time=end-start
        list_time["number_of_event_attachment"]=ex_time

def Javascript_rightclick_disabled(html, list_features, list_time):
    #global list_features
    if config["Javascript_Features"]["rightclick_disabled"] == "True":
        start=time.time()
        rightclick_disabled = 0
        if html:
            try:
                rightclick_disabled = 0
                #print(html.text.lower())
                if 'addEventListener(\'contextmenu\'' in html.html.lower():
                    rightclick_disabled = 1
            except Exception as e:
                logger.warning("exception: " + str(e))
                rightclick_disabled=-1
        list_features["rightclick_disabled"]=rightclick_disabled
        end=time.time()
        ex_time=end-start
        list_time["rightclick_disabled"]=ex_time

def Javascript_number_of_total_suspicious_features(list_features,list_time):
    if config["Javascript_Features"]["number_of_total_suspicious_features"] == "True":
        start=time.time()
        number_of_total_suspicious_features=0
        try:
            number_of_total_suspicious_features = list_features["number_of_exec"] + list_features["number_of_escape"] + list_features["number_of_eval"] + list_features["number_of_link"] +list_features["number_of_unescape"] + list_features["number_of_search"] \
            +list_features["rightclick_disabled"] + list_features["number_of_event_attachment"] + list_features["number_of_iframes_in_script"] + list_features["number_of_event_attachment"] + list_features["number_of_setTimeout"]
        except Exception as e:
            logger.warning("exception: " + str(e))
            number_of_total_suspicious_features=-1
        list_features["number_of_total_suspicious_features"]=number_of_total_suspicious_features
        end=time.time()
        ex_time=end-start
        list_time["number_of_total_suspicious_features"]=ex_time

def Email_Body_tfidf_emails(list_time):
    if config["Email_Body_Features"]["tfidf_emails"] == "True":
        start=time.time()
        Tfidf_matrix = Tfidf.tfidf_emails()
        end=time.time()
        ex_time=end-start
        list_time["tfidf_emails"]=ex_time
        return Tfidf_matrix

def Email_Header_Header_Tokenizer(list_time):
    if config["Email_Header_Features"]["Header_Tokenizer"] == "True":
        start=time.time()
        header_tokenizer=Tfidf.Header_Tokenizer()
        end=time.time()
        ex_time=end-start
        list_time["header_tokenizer"]=ex_time
        return header_tokenizer

def HTML_tfidf_websites(list_time):
    if config["HTML_Features"]["tfidf_websites"] == "True":
        start=time.time()
        Tfidf_matrix = Tfidf.tfidf_websites()
        end=time.time()
        ex_time=end-start
        list_time["tfidf_websites"]=ex_time
        return Tfidf_matrix

def extract_email_features(dataset_path, feature_list_dict, extraction_time_dict):
    data = list()
    corpus_data = read_corpus(dataset_path)
    data.extend(corpus_data)
    features_regex = re.compile(dataset_path + r"_features_?\d?.txt")
    ### for debugging purposes, not used in the pipeline
    try:
        list_files=os.listdir('.')
        count_feature_files=len(re.findall(features_regex,''.join(list_files)))
        logger.info("Total number of features files: {}".format(count_feature_files))
        features_output = dataset_path + "_feature_vector_" + str(count_feature_files+ 1) + ".txt"
    except Exception as e:
        features_output = dataset_path + "_feature_vector_error.txt"
        logger.warning("exception: " + str(e))
    ###
    corpus=[]
    for filepath in data:
        dict_features={}
        dict_time={}
        logger.info("===================")
        logger.info(filepath)
        email_features(filepath, dict_features, features_output, feature_list_dict, dict_time, extraction_time_dict, corpus)
        summary.write("filepath: {}\n\n".format(filepath))
        summary.write("features extracted for this file:\n")
        for feature in dict_time.keys():
            summary.write("{} \n".format(feature))
            summary.write("extraction time: {} \n".format(dict_time[feature]))
        summary.write("\n#######\n")
    count_files=len(feature_list_dict)
    return count_files, corpus

def extract_url_features(dataset_path, feature_list_dict, extraction_time_dict, Bad_URLs_List):
    data = list()
    corpus_data = read_corpus(dataset_path)
    data.extend(corpus_data)
    ## for debugging purposes, not used in the pipeline
    ###
    corpus=[]
    for filepath in data:
        # path="Data_Dump/URLs_Backup/"+str(ntpath.normpath(filepath).split('\\'))
        # features_regex=re.compile(path+r"_features_?\d?.txt")
        # try:
        #     list_files=os.listdir('.')
        #     count_feature_files=len(re.findall(features_regex,''.join(list_files)))
        #     logger.debug(count_feature_files)
        #     features_output=path+"_feature_vector_"+str(count_feature_files+ 1)+".txt"
        # except Exception as e:
        #     features_output=path+"_feature_vector_error.txt"
        #     logger.warning("exception: " + str(e))
        dict_features={}
        dict_time={}
        logger.info("===================")
        logger.info(filepath)
        #with open("Data_Dump/URLs_Training/features_url_training_legit.pkl",'ab') as feature_tracking:
        url_features(filepath, dict_features, feature_list_dict, dict_time, extraction_time_dict, corpus, Bad_URLs_List)
        summary.write("filepath: {}\n\n".format(filepath))
        summary.write("features extracted for this file:\n")
        for feature in dict_time.keys():
            summary.write("{} \n".format(feature))
            summary.write("extraction time: {} \n".format(dict_time[feature]))
        summary.write("\n#######\n")
    count_files=len(feature_list_dict)
    return count_files, corpus


def Extract_Features_Emails_Training():
    #summary=open(config["Summary"]["Path"],'w')
    start_time = time.time()
    logger.info("===============================================================")
    ### Training Features
    logger.info(">>>>> Feature extraction: Training Set >>>>>")
    dataset_path_legit_train=config["Dataset Path"]["path_legitimate_training"]
    dataset_path_phish_train=config["Dataset Path"]["path_phishing_training"]
    feature_list_dict_train=[]
    extraction_time_dict_train=[]
    labels_legit_train, data_legit_train=extract_email_features(dataset_path_legit_train, feature_list_dict_train, extraction_time_dict_train)
    labels_all_train, data_phish_train=extract_email_features(dataset_path_phish_train, feature_list_dict_train, extraction_time_dict_train)
    logger.debug(">>>>> Feature extraction: Training Set >>>>> Done ")

    logger.info(">>>>> Cleaning >>>>")
    logger.debug("feature_list_dict_train{}".format(len(feature_list_dict_train)))
    Cleaning(feature_list_dict_train)
    logger.debug(">>>>> Cleaning >>>>>> Done")
    labels_train=[]
    for i in range(labels_legit_train):
        labels_train.append(0)
    for i in range(labels_all_train-labels_legit_train):
        labels_train.append(1)
    corpus_train = data_legit_train + data_phish_train
    return feature_list_dict_train, labels_train, corpus_train

def Extract_Features_Emails_Testing():
    #summary=open(config["Summary"]["Path"],'w')
    start_time = time.time()
    logger.info("===============================================================")
    dataset_path_legit_test=config["Dataset Path"]["path_legitimate_testing"]
    dataset_path_phish_test=config["Dataset Path"]["path_phishing_testing"]
    feature_list_dict_test=[]
    extraction_time_dict_test=[]
    labels_legit_test, data_legit_test=extract_email_features(dataset_path_legit_test, feature_list_dict_test, extraction_time_dict_test)
    labels_all_test, data_phish_test=extract_email_features(dataset_path_phish_test, feature_list_dict_test, extraction_time_dict_test)
    logger.debug(">>>>> Feature extraction: Testing Set >>>>> Done ")
    logger.info(">>>>> Cleaning >>>>")
    logger.debug("feature_list_dict_test{}".format(len(feature_list_dict_test)))
    Cleaning(feature_list_dict_test)
    logger.debug(">>>>> Cleaning >>>>>> Done")
    labels_test=[]
    for i in range(labels_legit_test):
        labels_test.append(0)
    for i in range(labels_all_test-labels_legit_test):
        labels_test.append(1)
    corpus_test = data_legit_test + data_phish_test
    logger.info("--- %s final count seconds ---" % (time.time() - start_time))
    return feature_list_dict_test, labels_test, corpus_test
 

def Extract_Features_Urls_Training():
    #summary=open(config["Summary"]["Path"],'w')
    if config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
        start_time = time.time()
        logger.info("===============================================================")
        logger.info("===============================================================")
        logger.info(">>>>> Feature extraction: Training Set >>>>>")
        dataset_path_legit_train=config["Dataset Path"]["path_legitimate_training"]
        dataset_path_phish_train=config["Dataset Path"]["path_phishing_training"]
        feature_list_dict_train=[]
        feature_list_dict_train2=[]
        extraction_time_dict_train=[]
        Bad_URLs_List=[]
        t0 = time.time()
        #with open("Data_Dump/URLs_Training/features_url_training_legit.pkl",'ab') as feature_tracking:
        labels_legit_train, data_legit_train=extract_url_features(dataset_path_legit_train, feature_list_dict_train, extraction_time_dict_train, Bad_URLs_List)          
        labels_all_train, data_phish_train=extract_url_features(dataset_path_phish_train, feature_list_dict_train, extraction_time_dict_train, Bad_URLs_List)
        logger.info("Feature extraction time is: {}s".format(time.time() - t0)) 
        logger.debug(">>>>> Feature extraction: Training Set >>>>> Done ")
        Cleaning(feature_list_dict_train)
        logger.debug(">>>>> Cleaning >>>>>> Done")
        #logger.info("Number of bad URLs in training dataset: {}".format(len(Bad_URLs_List)))

        labels_train=[]
        for i in range(labels_legit_train):
            labels_train.append(0)
        for i in range(labels_all_train-labels_legit_train):
            labels_train.append(1)

        #logger.info("\nfeature_list_dict_train2: {}\n".format(feature_list_dict_train2))
        corpus_train = data_legit_train + data_phish_train
#
#        #logger.info("--- %s final count seconds ---" % (time.time() - start_time))
        return feature_list_dict_train, labels_train, corpus_train

        print("--- %s final count seconds ---" % (time.time() - start_time))
   
def Extract_Features_Urls_Testing():
    start_time = time.time()
    logger.info(">>>>> Feature extraction: Testing Set")
    dataset_path_legit_test=config["Dataset Path"]["path_legitimate_testing"]
    dataset_path_phish_test=config["Dataset Path"]["path_phishing_testing"]
    feature_list_dict_test=[]
    extraction_time_dict_test=[]
    Bad_URLs_List=[]
    labels_legit_test, data_legit_test=extract_url_features(dataset_path_legit_test, feature_list_dict_test, extraction_time_dict_test, Bad_URLs_List)
    labels_all_test, data_phish_test=extract_url_features(dataset_path_phish_test, feature_list_dict_test, extraction_time_dict_test, Bad_URLs_List)
    logger.debug(">>>>> Feature extraction: Testing Set >>>>> Done ")
    logger.info(">>>>> Cleaning >>>>")
    logger.debug("feature_list_dict_test{}".format(len(feature_list_dict_test)))
    Cleaning(feature_list_dict_test)
    logger.debug(">>>>> Cleaning >>>>>> Done")
    #logger.info("Number of bad URLs in training dataset: {}".format(len(Bad_URLs_List)))
    labels_test=[]
    for i in range(labels_legit_test):
        labels_test.append(0)
    for i in range(labels_all_test-labels_legit_test):
        labels_test.append(1)

    corpus_test = data_legit_test + data_phish_test
    logger.info("--- %s final count seconds ---" % (time.time() - start_time))
    return feature_list_dict_test, labels_test, corpus_test
   

    
