import copy
import csv
import email as em
import ntpath
import os
import os.path
import pickle
import re
import string
import sys
import time
import traceback
from itertools import groupby
from urllib.parse import urlparse

import nltk
import numpy as np
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from scipy import stats
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer


from . import Features
from .utils import Globals

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



############################




############################

def read_corpus(path):

  # assumes a flat directory structure 
    files = filter(lambda x: x.endswith('.txt'), os.listdir(path))
    paths = map(lambda x: os.path.join(path,x), files)
    return list(paths)


############################


############################

def read_alexa(path):
    reader = csv.DictReader(open(path))
    result = {}
    for row in reader:
        result[row["domain"]] = row["rank"]
    return result
############################


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
    #Globals.summary.open(Globals.config["Summary"]["Path"],'w')
    Globals.summary.write("\n\n###### List of Preprocessing steps:\n")
    #Array_Features=Sparse_Matrix_Features.toarray()
    X_array=X.toarray()
    #X_test_array=X_test.toarray()
    #scaled_Array_Features=preprocessing.scale(Sparse_Matrix_Features) # Center data with the mean and then scale it using the std deviation
    #other method that keeps the model for testing
    #if Globals.config["Preprocessing"]["mean_scaling"] == "True":
    #    X_train=mean_scaling(X_train)
    #    X_test=mean_scaling(X_test)
    #    Globals.summary.write("\n Scaling using the mean.\n")
    #    print("Preprocessing: Mean_scaling")
    #    return X_train, X_test
    #    # return the scaler for testing data
    #    # Use min max to scale data because it's robust to very small standard deviations of features and preserving zero
    if Globals.config["Preprocessing"]["min_max_scaling"] == "True":
        X=min_max_scaling(X_array)
        #X_test=min_max_scaling(X_test_array)
        Globals.summary.write("\n Scaling using the min and max.\n")
        Globals.logger.info("Preprocessing: min_max_scaling")
        return X
        # use abs value to scale
    #elif Globals.config["Preprocessing"]["abs_scaler"] == "True":
    #    X_train=abs_scaler(X_train)
    #    X_test=abs_scaler(X_test)
    #    Globals.summary.write("\n Scaling using the absolute value.\n")
    #    print("Preprocessing: abs_scaler")
    #    return X_train, X_test
    #    #normalize the data???
    #elif Globals.config["Preprocessing"]["normalize"] == "True":
    #    X_train = normalizer(X_train)
    #    X_test = normalizer(X_test)
    #    Globals.summary.write("\n Normalizing.\n")
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
                Globals.logger.debug("Value of {} changed from {} to {}".format(key,original,item[key]))


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










# sys.setdefaultencoding('utf-8')

# get filename from path
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)



