from sklearn import svm
from sklearn import datasets
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
from imblearn.datasets import make_imbalance
from imblearn.under_sampling import RandomUnderSampler,CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.datasets import load_svmlight_file
from imblearn.metrics import geometric_mean_score
#from imblearn.metrics import Balanced_accuracy_score
#import User_options
import sklearn
import configparser
#from collections import deque
import Features
import tensorflow as tf
import logging

logger = logging.getLogger('root')

config=configparser.ConfigParser()
config.read('Config_file.ini')

def Confusion_matrix(y_test, y_predict):
		confusion_matrix=sklearn.metrics.confusion_matrix(y_test, y_predict, labels=[0,1])
		tn, fp, fn, tp=confusion_matrix.ravel()
		logger.info("Confusion Matrix (TN, FP, FN, TP):({}, {}, {}, {})".format(tn, fp, fn, tp))
		return ([tn, fp, fn, tp])

def Confusion_matrix2(y_test, y_predict):
		sess = tf.Session()
		with sess.as_default():
			y_test=y_test.eval()
			y_predict=y_predict.eval()
		confusion_matrix=sklearn.metrics.confusion_matrix(y_test, y_predict)
		tn, fp, fn, tp=confusion_matrix.ravel()
		logger.info("Confusion Matrix (TN, FP, FN, TP):({}, {}, {}, {})".format(tn, fp, fn, tp))


def Matthews_corrcoef(y_test, y_predict):
		Mcc=sklearn.metrics.matthews_corrcoef(y_test, y_predict)
		logger.info("Matthews_CorrCoef: {}".format(Mcc))
		return (Mcc)
		#return Mcc

def ROC_AUC(y_test, y_predict):
		ROC_AUC=sklearn.metrics.roc_auc_score(y_test, y_predict)
		logger.info("ROC_AUC: {}".format(ROC_AUC))
		return (ROC_AUC)
		#return ROC_AUC

def Precision(y_test, y_predict):
		precision=sklearn.metrics.precision_score(y_test, y_predict)
		logger.info("Precision: {}".format(precision))
		return (precision)
		#return precision

def Recall(y_test, y_predict):
		recall=sklearn.metrics.recall_score(y_test, y_predict)
		logger.info("Recall: {}".format(recall))
		return recall
		#return Recall

def F1_score(y_test, y_predict):
		f1_score=sklearn.metrics.f1_score(y_test, y_predict)
		logger.info("F1_score: {}".format(f1_score))
		return f1_score
		#return F1_score

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def Cross_validation(clf, X, y):
		scoring = {'tp' : make_scorer(tp), 'tn' : make_scorer(tn),
		           'fp' : make_scorer(fp), 'fn' : make_scorer(fn)}
		cv_results = cross_validate(clf, X, y, cv=10, scoring=scoring, verbose=1, n_jobs=1, return_train_score=False)
		#scores = cross_validate(clf, X, y, cv=10, verbose=1, n_jobs=-1,)
		#conf_mat = confusion_matrix(y, y_predict)
		logger.info("10 fold Cross_Validation: {}".format(cv_results))

def Homogenity(y_test,y_predict):
		homogenity=sklearn.metrics.homogeneity_score(y_test,y_predict)
		logger.info("Homogenity: {}".format(homogenity))

def Completeness(y_test,y_predict):
		completeness=sklearn.metrics.completeness_score(y_test,y_predict)
		logger.info("Completeness: {}".format(completeness))

def V_measure(y_test,y_predict):
		v_measure=sklearn.metrics.v_measure_score(y_test,y_predict)
		logger.info("V_measure: {}".format(v_measure))

def Geomteric_mean_score(y_test,y_predict):
		g_mean=geometric_mean_score(y_test,y_predict)
		logger.info("G_mean: {}".format(g_mean))
		return g_mean

def Balanced_accuracy_score(y_test,y_predict):
		b_accuracy=sklearn.metrics.balanced_accuracy_score(y_test,y_predict)
		logger.info("Balanced_accuracy_score: {}".format(b_accuracy))
		return b_accuracy

def eval_metrics(clf, y_test, y_predict):
	summary=Features.summary
	summary.write("\n\nEvaluation metrics used:\n")
	summary.write("\n\n Supervised metrics:\n")
	eval_metrics_dict = {}
	if config["Evaluation Metrics"]["Confusion_matrix"] == "True":
		cm = Confusion_matrix(y_test, y_predict)
		eval_metrics_dict['CM'] = cm
		summary.write("Confusion_matrix\n")
	if config["Evaluation Metrics"]["Matthews_corrcoef"] == "True":
		mcc = Matthews_corrcoef(y_test, y_predict)
		eval_metrics_dict['MCC'] = mcc
		summary.write("Matthews_corrcoef\n")
	if config["Evaluation Metrics"]["ROC_AUC"] == "True":
		roc_auc = ROC_AUC(y_test, y_predict)
		eval_metrics_dict['ROC_AUC'] = roc_auc
		summary.write("ROC_AUC\n")
	if config["Evaluation Metrics"]["Precision"] == "True":
		precision = Precision(y_test, y_predict)
		eval_metrics_dict['Precision'] = precision
		summary.write("Precision\n")
	if config["Evaluation Metrics"]["Recall"] == "True":
		recall = Recall(y_test, y_predict)
		eval_metrics_dict['Recall'] = recall
		summary.write("Recall\n")
	if config["Evaluation Metrics"]["F1_score"] == "True":
		f1_score = F1_score(y_test, y_predict)
		eval_metrics_dict['F1_score'] = f1_score
		summary.write("F1_score\n")
	#if config["Evaluation Metrics"]["Cross_validation"] == "True":
	#	Cross_validation(clf, X, y)
	#	summary.write("Cross_validation\n")
	if config["Evaluation Metrics"]["Geomteric_mean_score"] == "True":
		gmean = Geomteric_mean_score(y_test,y_predict)
		eval_metrics_dict['Gmean'] = gmean
		summary.write("Geomteric_mean_score\n")
	if config["Evaluation Metrics"]["Balanced_accuracy_score"] == "True":
		accuracy = Balanced_accuracy_score(y_test,y_predict)
		eval_metrics_dict['accuracy'] = accuracy
		summary.write("Balanced_accuracy_score\n")
	#	# write results to summary
	if config["Classification"]["Attack Features"] == "True":
		logger.debug("Original Labels: {}".format(y_test))
		logger.debug("New Labels: {}".format(y_predict))
	return (eval_metrics_dict)

def eval_metrics_cluster(y_test, y_predict):
	summary=Features.summary
	summary.write("\n\nEvaluation metrics used:\n")
	summary.write("\n\n clustering metrics:\n")
	eval_metrics_dict_cluster = {}
	if config["Evaluation Metrics"]["Homogenity"] == "True":
		homogeneity_score = Homogenity(y_test, y_predict)
		eval_metrics_dict_cluster['Homogenity'] = homogeneity_score
		summary.write("Homogenity\n")
	if config["Evaluation Metrics"]["Completeness"] == "True":
		completeness_score = Completeness(y_test, y_predict)
		eval_metrics_dict_cluster['Completeness'] = completeness_score
		summary.write("Completeness\n")
	if config["Evaluation Metrics"]["V_measure"] == "True":
		v_measure_score = V_measure(y_test,y_predict)
		eval_metrics_dict_cluster['V_measure'] = v_measure_score
		summary.write("V_measure\n")
	return (eval_metrics_dict_cluster)
