import os
import sys
import ast
from bs4 import BeautifulSoup, Comment
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.externals import joblib
import pickle
import argparse
import re
import Features_Support
from scipy.sparse import hstack
from langdetect import detect
import langdetect

# prog = re.compile("('[a-zA-Z0-9_\-\. ]*':\"'[a-zA-Z0-9_\-\. ]*'\")|('[a-zA-Z0-9_\-\. ]*':\"[a-zA-Z0-9_\-\. ]*\")|('[a-zA-Z0-9_\-\. ]*':[0-9\.[0-9]*)|('[a-zA-Z0-9_\-\. ]*':*)")


parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('--html_content', type=str, required=True,
                    help='path to the html pkl file.')
parser.add_argument('--features', type=str, required=False,
                    help='path to the feature sparse matrix unprocessed file.')
parser.add_argument('--labels', type=str, required=True,
                    help='path to the label file.')
parser.add_argument('--dataset_name', type=str, required=False,
                    help=' name of dataset.')
parser.add_argument('--output_dir', type=str, required=False,
                    help='directory to store the features.')

args = parser.parse_args()

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    elif isinstance(element, Comment):
        return False
    return True
def html_feature_extractor(html):
    global counter
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except NotImplementedError:
        soup = BeautifulSoup(html, 'html5lib')
    #soup = BeautifulSoup(html, 'html5lib')
    body = []    
    data = soup.findAll(text=True)
    result = filter(visible, data)
    regex = re.compile('<[a-z][\s\S]*>', re.IGNORECASE)
    try:
        for res in result:
            sentence = str(res)
            sentence = ' '.join(sentence.split())
            sentence_cleaned = sentence.strip().rstrip('\n')
            if sentence_cleaned == '':
                continue
            if regex.match(sentence_cleaned):
                soup_t = BeautifulSoup(sentence_cleaned, 'html5lib')
                for temp in filter(visible, soup_t.findAll(text=True)): 
                    temp = ' '.join(temp.split()).strip().rstrip('\n')
                    if temp == '':
                        continue
                    try:
                        if detect(sentence_cleaned) == 'en':
                            body.append(temp)
                    except langdetect.lang_detect_exception.LangDetectException as ex:
                        pass
            else:
                try:
                    if detect(sentence_cleaned) == 'en':
                        body.append(sentence_cleaned)
                except langdetect.lang_detect_exception.LangDetectException:
                    pass
    except UnicodeEncodeError as e:
        raise e
    return body

def binormal_separation(corpus):
    y_train=joblib.load(args.labels)
    vocab = []
    vectorizer=CountVectorizer(analyzer='word', ngram_range=(1,1), min_df = 5, stop_words = 'english')
    analyzer = vectorizer.build_analyzer()
    new_corpus= []
    input_dict = {}
    input_dict['phish'] = []
    input_dict['legit'] = []
    phish_counter = 0
    legit_counter = 0
    for i, document in enumerate(corpus):
        if y_train[i] == 0:
            input_dict['legit'].append(analyzer(document))
            legit_counter += 1
        if y_train[i] == 1:
            input_dict['phish'].append(analyzer(document))
            phish_counter += 1

    print("legit: {:d}, phish: {:d}".format(legit_counter, phish_counter))
    from DocumentFeatureSelection import interface
    rankings = interface.run_feature_selection(input_dict, method='bns', use_cython=True, is_use_cache=True).convert_score_matrix2score_record()
    joblib.dump(rankings, "binormal.pkl")
    for i, item in enumerate(rankings):
        if i< 100:
            vocab.append(item['feature'])
    print(vocab)
    return vocab

def website_tfidf(run_binormal_separation=False):
        corpus=convert_from_pkl_to_text(args.html_content)
        print("length of list of html content (rows in tfidf matrix): {}".format(len(corpus)))
        vocab = None
        if run_binormal_separation:
            vocab = binormal_separation(corpus)
        tfidf_matrix=Tfidf_Vectorizer(corpus, vocabulary=vocab)
        if args.features:
                X_features = joblib.load(args.features)
                return Combine_Matrix(X_features,tfidf_matrix)

def url_tokenizer(input):
    return re.split('[^a-zA-Z]', input) 

def url_tfidf(word=True, X_input=None):
    corpus=convert_from_pkl_to_text(args.html_content)
    print("length of list of URLs (rows in tfidf matrix): {}".format(len(corpus)))
    if word:
        tfidf_matrix=Tfidf_Vectorizer(corpus, analyzer='word', tokenizer=url_tokenizer, idf=False)
    else:
        tfidf_matrix=Tfidf_Vectorizer(corpus, analyzer='char', idf=False)
    if X_input:
        return Combine_Matrix(X_input,tfidf_matrix)


def convert_from_pkl_to_text(input_file, url=False):
    text=''
    first_line=1
    corpus=[]
    with open(input_file, 'rb') as f:
        try:
            while(True):
                data=joblib.load(f)
                if url and data.startswith("URL: "):
                    corpus.append(data.split(":", 1)[1])
                elif not url and not data.startswith("URL: "):
                    corpus.append("\n".join(html_feature_extractor(data)))
        except (EOFError):
            pass
    return corpus


def Tfidf_Vectorizer(corpus, analyzer='word', tokenizer=None, idf=True, vocabulary=None):
    if idf:
        vectorizer=TfidfVectorizer(analyzer=analyzer, ngram_range=(1,1), tokenizer=tokenizer,
                                    min_df = 5, stop_words = 'english', sublinear_tf=True, vocabulary=vocabulary)
    else:
                vectorizer=CountVectorizer(analyzer=analyzer, ngram_range=(1,1), tokenizer=tokenizer,
                        min_df = 5, stop_words = 'english', vocabulary=vocabulary)
    tfidf_matrix=vectorizer.fit_transform(corpus)
    joblib.dump(tfidf_matrix, os.path.join(args.output_dir,args.dataset_name+'_tfidf_matrix_combined.pkl'))
    return tfidf_matrix

def Combine_Matrix(m1, m2):
    print(m1.shape, m2.shape)
    X=hstack([m1, m2])
    X=Features_Support.Preprocessing(X)
    return X

if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #X = url_tfidf(False, url_tfidf(True, website_tfidf()))

    X = website_tfidf(True)
    if args.features:
        joblib.dump(X, os.path.join(args.output_dir,args.dataset_name+"_Features_with_Tfidf_processed.pkl"))

