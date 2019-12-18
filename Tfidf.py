from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import Features_Support
#import User_options
import Download_url
import configparser
#from collections import deque
import logging

logger = logging.getLogger('root')

config=configparser.ConfigParser()
config.read('Config_file.ini')

## Build the corpus from both the datasets
def build_corpus():
	data=list()
	path=config["Dataset Path"]["path_legit_email"]
	corpus_data_legit = Features_Support.read_corpus(path)
	logger.info("Corpus Data legit: >>>>>>>>>>>>>>> " + str(len(corpus_data_legit)))
	data.extend(corpus_data_legit)
	#for path in config["Dataset Path"][""]path_phish_email:
	path = config["Dataset Path"]["path_phish_email"]
	corpus_data_phish = Features_Support.read_corpus(path)
	logger.info("Corpus Data phish: >>>>>>>>>>>>>>> " + str(len(corpus_data_phish)))
	data.extend(corpus_data_phish)
	return data


def tfidf_training(corpus):
	tf= TfidfVectorizer(analyzer='word', ngram_range=(1,1),
                     min_df = 0, stop_words = 'english', sublinear_tf=True)		
	tfidf_matrix = tf.fit_transform(corpus)
	return tfidf_matrix, tf

def tfidf_testing(corpus):
	tfidf_matrix = tf.transform(corpus)
	return tfidf_matrix

def Header_Tokenizer(corpus):
	# corpus=[]
	# data=build_corpus()
	data=corpus
	#for filepath in data:
	#	try:
	#		print(filepath)
	#		with open(filepath,'r', encoding = "ISO-8859-1") as f:
	#			email=f.read()
	#			header=Features_Support.extract_header(email)
	#			corpus.append(header)
	#	except Exception as e:
	#		print("exception: " + str(e))
	cv= CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words='english',
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)
	header_tokenizer = cv.fit_transform(corpus)
	return header_tokenizer	


if __name__ == '__main__':
	matrix=tfidf_website()
	logger.info(matrix)
