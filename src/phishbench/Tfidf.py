from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_training(corpus):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),
                         min_df=0, stop_words='english', sublinear_tf=True)
    tfidf_matrix = tf.fit_transform(corpus)
    return tfidf_matrix, tf


def tfidf_testing(corpus, tidf_vectorizer):
    tfidf_matrix = tidf_vectorizer.transform(corpus)
    return tfidf_matrix


def Header_Tokenizer(corpus):
    cv = CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                         encoding='utf-8', input='content',
                         lowercase=True, max_df=1.0, max_features=None, min_df=1,
                         ngram_range=(1, 1), preprocessor=None, stop_words='english',
                         strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                         tokenizer=None, vocabulary=None)
    header_tokenizer = cv.fit_transform(corpus)
    return header_tokenizer
