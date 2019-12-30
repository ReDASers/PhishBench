import os
import sys
from Classifiers import classifiers
from Classifiers import fit_MNB
import Features
import Features_Support
import Feature_Selection
import Imbalanced_Dataset
from sklearn.externals import joblib
#import User_options
import re
#from Classifiers_test import load_dataset
import configparser
import Tfidf
from scipy.sparse import hstack
#from collections import deque
import logging
import argparse
from sklearn.datasets import dump_svmlight_file


parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("-o", "--output_input_dir", help="Output/input directory to read features or dump extracted features",
                    type=str, default="Data_Dump")
parser.add_argument("-c", "--ignore_confirmation", help="does not wait or user's confirmation",
                    action="store_true")

args = parser.parse_args()

config=configparser.ConfigParser()
config.read('Config_file.ini')

def setup_logger():
    # create formatter
    # formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
    # create console handler and set level to debug
    handler = logging.StreamHandler()
    # add formatter to handler
    handler.setFormatter(formatter)
    # create logger
    logger = logging.getLogger('root')
    if args.verbose:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)

setup_logger()
logger = logging.getLogger('root')

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def Confirmation():
    print("##### Review of Options:")
    if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
        print("extract_features_emails = {}".format(config["Email or URL feature Extraction"]["extract_features_emails"]))
    elif config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
        print("extract_features_urls = {}".format(config["Email or URL feature Extraction"]["extract_features_urls"]))

    print("###Paths to datasets:")
    print("Legitimate Dataset (Training): {}".format(config["Dataset Path"]["path_legitimate_training"]))
    print("Phishing Dataset (Training):: {}".format(config["Dataset Path"]["path_phishing_training"]))
    print("Legitimate Dataset (Testing): {}".format(config["Dataset Path"]["path_legitimate_testing"]))
    print("Phishing Dataset (Testing): {}".format(config["Dataset Path"]["path_phishing_testing"]))

    print("\nRun Feature Ranking Only: {}".format(config["Feature Selection"]["Feature Ranking Only"]))
    if config["Extraction"]["feature extraction"]=="True":
        print("\nRun the Feature Extraction: {}".format(config["Extraction"]["feature extraction"]))
        print("\nFeature Extraction for Training Data: {}".format(config["Extraction"]["training dataset"]))
        print("\nFeature Extraction for Testing Data: {}".format(config["Extraction"]["testing dataset"]))
    else:
        print("\nRun the Feature Extraction: {}".format(config["Extraction"]["feature extraction"]))
    print("\nRun the classifiers: {}".format(config["Classification"]["Running the classifiers"]))
    print("\n")
    if args.ignore_confirmation:
        answer = True
    else:
        answer = query_yes_no("Do you wish to continue?")
    return answer

def load_dataset(load_train=True, load_test=False):
    y_test = None
    X_test = None
    X_train = None
    y_train = None
    vectorizer_train = None
    vectorizer_test = None
    if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
        email_train_dir = os.path.join(args.output_input_dir, "Emails_Training")
        vectorizer_train=joblib.load(os.path.join(email_train_dir, "vectorizer.pkl"))
        if load_train:       
            X_train=joblib.load(os.path.join(email_train_dir, "X_train.pkl"))
            y_train=joblib.load(os.path.join(email_train_dir, "y_train.pkl"))
            
        try:
            if load_test:
                if config["Classification"]["Attack Features"] == "True":
                    with open(os.path.join(email_train_dir,"features.txt"),'r') as f:
                          dict_test=eval(f.read())
                    X_test=vectorizer_train.fit_transform(dict_test)
                    y_test=joblib.load(os.path.join(email_train_dir, "y_test.pkl"))
                else:
                    email_test_dir = os.path.join(args.output_input_dir, "Emails_Testing")
                    vectorizer_test=joblib.load(os.path.join(email_test_dir, "vectorizer.pkl"))
                    X_test=joblib.load(os.path.join(email_test_dir, "X_test.pkl"))
                    y_test=joblib.load(os.path.join(email_test_dir, "y_test.pkl"))
        except FileNotFoundError as ex:
            logger.warn("Test files not found {}".format(ex))

    elif config["Email or URL feature Extraction"]["extract_features_URLs"] == "True":
        url_train_dir = os.path.join(args.output_input_dir, "URLs_Training")
        url_test_dir = os.path.join(args.output_input_dir, "URLs_Testing")
        vectorizer_train= joblib.load(os.path.join(url_train_dir, "vectorizer.pkl"))
        if load_train:
            X_train=joblib.load(os.path.join(url_train_dir, "X_train.pkl"))
            y_train=joblib.load(os.path.join(url_train_dir, "y_train.pkl"))
        try:
            if load_test:
                if config["Classification"]["Attack Features"] == "True":
                    with open(os.path.join(url_train_dir,"features.txt"),'r') as f:
                          dict_test=eval(f.read())
                    X_test=vectorizer_test.fit_transform(dict_test)
                    y_test=joblib.load(os.path.join(url_train_dir, "y_test.pkl"))
                else:
                    X_test=joblib.load(os.path.join(url_test_dir, "X_test.pkl"))
                    y_test=joblib.load(os.path.join(url_test_dir, "y_test.pkl"))
                    vectorizer_test= joblib.load(os.path.join(url_test_dir, "vectorizer.pkl"))
        except FileNotFoundError as ex:
            logger.warn("Test files not found {}".format(ex))

    return X_train, y_train, X_test, y_test, vectorizer_train, vectorizer_test

def main():
    Feature_extraction=False #flag for feature extraction
    flag_training=False
    # Feature dumping and loading methods
    # flag_saving_pickle=config["Features Format"]["Pikle"]
    # flag_saving_svmlight=config["Features Format"]["Svmlight format"]


### Feature ranking only
    ranking_dir = os.path.join(args.output_input_dir, "Feature_Ranking")
    email_train_dir = os.path.join(args.output_input_dir, "Emails_Training")
    email_test_dir = os.path.join(args.output_input_dir, "Emails_Testing")
    url_train_dir = os.path.join(args.output_input_dir, "URLs_Training")
    url_test_dir = os.path.join(args.output_input_dir, "URLs_Testing")
    if config["Feature Selection"]["Feature Ranking Only"]=='True':
        if config["Extraction"]["feature extraction"] == "True":
            if not os.path.exists(ranking_dir):
                os.makedirs(ranking_dir)
            if config["Email or URL feature Extraction"]["extract_features_emails"] == "True": 
                if not os.path.exists(email_train_dir):
                    os.makedirs(email_train_dir)
                (feature_list_dict_train, y, corpus)=Features.Extract_Features_Emails_Training()
                X, vectorizer=Features_Support.Vectorization_Training(feature_list_dict_train)
                X=Features_Support.Preprocessing(X)
                joblib.dump(vectorizer,os.path.join(email_train_dir, "vectorizer.pkl"))

            elif config["Email or URL feature Extraction"]["extract_features_URLs"] == "True":
                if not os.path.exists(url_train_dir):
                    os.makedirs(url_train_dir)
                (feature_list_dict_train, y, corpus_train)=Features.Extract_Features_Urls_Training()
                X, vectorizer=Features_Support.Vectorization_Training(feature_list_dict_train)
                X=Features_Support.Preprocessing(X)
                joblib.dump(vectorizer,os.path.join(url_train_dir, "vectorizer.pkl"))
        
        else: 
            X, y, X_test, y_test, vectorizer_train, vectorizer_test = load_dataset()
            #feature_list_dict_train=vectorizer_train.inverse_transform(X)

        logger.info("Select Best Features ######")
        k = int(config["Feature Selection"]["number of best features"])
        #X, selection = Feature_Selection.Select_Best_Features_Training(X, y, k)
        X, selection = Feature_Selection.Feature_Ranking(X, y,k)
        if config["Email or URL feature Extraction"]["extract_features_emails"] == "True": 
            joblib.dump(selection, os.path.join(email_train_dir, "selection.pkl"))
        elif config["Email or URL feature Extraction"]["extract_features_URLs"] == "True":
            joblib.dump(selection, os.path.join(url_train_dir, "selection.pkl"))
### Email FEature Extraction
    elif config["Extraction"]["Feature Extraction"]=='True':
        Feature_extraction=True
        if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
            if config["Extraction"]["Training Dataset"] == "True":
                # Create Data Dump directory if doesn't exist
                if not os.path.exists(email_train_dir):
                    os.makedirs(email_train_dir)
                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_train, y_train, corpus_train)=Features.Extract_Features_Emails_Training()
                # Tranform the list of dictionaries into a sparse matrix
                X_train, vectorizer=Features_Support.Vectorization_Training(feature_list_dict_train)
                # Save model for vectorization
                joblib.dump(vectorizer, os.path.join(email_train_dir, "vectorizer.pkl"))
                # Add tfidf if the user marked it as True
                if config["Email_Features"]["tfidf_emails"] == "True":
                    logger.info("tfidf_emails_train ######")
                    Tfidf_train, tfidf_vectorizer=Tfidf.tfidf_training(corpus_train)
                    X_train=hstack([X_train, Tfidf_train])
                    # Save tfidf model
                    joblib.dump(tfidf_vectorizer, os.path.join(email_train_dir, "tfidf_vectorizer.pkl"))
                
                # Use Min_Max_scaling for prepocessing the feature matrix
                X_train=Features_Support.Preprocessing(X_train)

                # feature ranking
                if config["Feature Selection"]["select best features"]=="True":
                    #k: Number of Best features
                    logger.info("Select Best Features ######")
                    k = int(config["Feature Selection"]["number of best features"])
                    #X_train, selection = Feature_Selection.Select_Best_Features_Training(X_train, y_train, k)
                    X_train, selection = Feature_Selection.Feature_Ranking(X_train, y_train,k)
                    # dump selection model
                    joblib.dump(selection, os.path.join(email_train_dir, "selection.pkl"))
                    logger.info("### Feature Ranking and Selection for Training Done!")
                
                
                # Train Classifiers on imbalanced dataset
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_train, y_train=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_train, y_train)

                # Fit classifier MNB
                #fit_MNB(X_train, y_train)
                # Save features for training dataset
                #dump_svmlight_file(X_train,y_train,"Data_Dump/Emails_Training/Feature_Matrix.txt")
                joblib.dump(X_train, os.path.join(email_train_dir, "X_train.pkl"))
                joblib.dump(y_train, os.path.join(email_train_dir, "y_train.pkl"))

                # flag to mark if training was done
                flag_training=True
                logger.info("Feature Extraction for training dataset: Done!")

            if config["Extraction"]["Testing Dataset"] == "True":
                # if training was done in another instance of the plaform then load the necessary files
                if flag_training==False:
                    #X_train=joblib.load(os.path.join(email_train_dir, "X_train.pkl"))
                    #y_train=joblib.load(os.path.join(email_train_dir, "y_train.pkl"))
                    vectorizer=joblib.load(os.path.join(email_train_dir, "vectorizer.pkl"))
                    
                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_test, y_test, corpus_test)=Features.Extract_Features_Emails_Testing()
                # Tranform the list of dictionaries into a sparse matrix
                X_test=Features_Support.Vectorization_Testing(feature_list_dict_test, vectorizer)
                
                # Add tfidf if the user marked it as True
                if config["Email_Features"]["tfidf_emails"] == "True":
                    tfidf_vectorizer=joblib.load(os.path.join(email_train_dir, "tfidf_vectorizer.pkl"))
                    logger.info("tfidf_emails_train ######")
                    Tfidf_test=Tfidf.tfidf_testing(corpus_test, tfidf_vectorizer)
                    X_test=hstack([X_test, Tfidf_test])
                
                # Use Min_Max_scaling for prepocessing the feature matrix
                X_test=Features_Support.Preprocessing(X_test)

                # feature ranking
                if config["Feature Selection"]["select best features"]=="True":
                    #k: Number of Best features
                    selection=joblib.load(os.path.join(email_train_dir, "selection.pkl"))
                    k = int(config["Feature Selection"]["number of best features"])
                    X_test = Feature_Selection.Select_Best_Features_Testing(X_test, selection, k, feature_list_dict_test)
                    logger.info("### Feature Ranking and Selection for Training Done!")
                
                # Train Classifiers on imbalanced dataset
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_test, y_test=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_test, y_test)

                #Dump Testing feature matrix with labels
                if not os.path.exists(email_test_dir):
                    os.makedirs(email_test_dir)
                joblib.dump(X_test, os.path.join(email_test_dir, "X_test.pkl"))
                joblib.dump(y_test, os.path.join(email_test_dir, "y_test.pkl"))
                if config["Extraction"]["Dump Features txt"] == "True":
                    joblib.dump(feature_list_dict_train,os.path.join(url_train_dir, "Features.txt"))
                logger.info("Feature Extraction for testing dataset: Done!")
            else:
                X_test = None
                y_test = None

######## URL feature extraction
        elif config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
            if config["Extraction"]["Training Dataset"] == "True":
                # Create directory to store dada
                if not os.path.exists(url_train_dir):
                    os.makedirs(url_train_dir)

                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_train, y_train, corpus_train)=Features.Extract_Features_Urls_Training()
                
                # Tranform the list of dictionaries into a sparse matrix
                X_train, vectorizer=Features_Support.Vectorization_Training(feature_list_dict_train)
                # Dump vectorizer
                joblib.dump(vectorizer, os.path.join(url_train_dir, "vectorizer.pkl"))
                joblib.dump(X_train, os.path.join(url_train_dir, "X_train_unprocessed.pkl"))
                # Add tfidf if the user marked it as True
                if config["HTML_Features"]["tfidf_websites"] == "True":
                    logger.info("Extracting TFIDF features for training websites ###### ######")
                    Tfidf_train, tfidf_vectorizer=Tfidf.tfidf_training(corpus_train)
                    joblib.dump(Tfidf_train, os.path.join(url_train_dir, "tfidf_features.pkl"))
                    X_train=hstack([X_train, Tfidf_train])
                    #dump tfidf vectorizer
                    joblib.dump(tfidf_vectorizer, os.path.join(url_train_dir, "tfidf_vectorizer.pkl"))
                
                joblib.dump(X_train, os.path.join(url_train_dir, "X_train_unprocessed_with_tfidf.pkl"))
                # Use Min_Max_scaling for prepocessing the feature matrix
                X_train=Features_Support.Preprocessing(X_train)
                joblib.dump(X_train, os.path.join(url_train_dir, "X_train_processed.pkl"))

                # Feature Selection
                if config["Feature Selection"]["select best features"]=="True":
                    #k: Number of Best features
                    k = int(config["Feature Selection"]["number of best features"])
                    X_train, selection = Feature_Selection.Feature_Ranking(X_train, y_train,k)
                    #Dump model
                    joblib.dump(selection, os.path.join(url_train_dir, "selection.pkl"))
                    joblib.dump(X_train, os.path.join(url_train_dir, "X_train_processed_best_features.pkl"))

                # Train Classifiers on imbalanced dataset
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_train, y_train=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_train, y_train)
                # dump features and labels and vectorizers

                joblib.dump(X_train, os.path.join(url_train_dir, "X_train.pkl"))
                joblib.dump(y_train, os.path.join(url_train_dir, "y_train.pkl"))
                if config["Extraction"]["Dump Features txt"] == "True":
                    joblib.dump(feature_list_dict_train,os.path.join(url_train_dir, "Features.txt"))
                # flag to mark if training was done
                flag_training=True
                logger.info("Feature Extraction for training dataset: Done!")

            if config["Extraction"]["Testing Dataset"] == "True":
                # if training was done in another instance of the plaform then load the necessary files
                if flag_training==False:
                    X_train=joblib.load(os.path.join(url_train_dir, "X_train.pkl"))
                    y_train=joblib.load(os.path.join(url_train_dir, "y_train.pkl"))
                    vectorizer=joblib.load(os.path.join(url_train_dir, "vectorizer.pkl"))
                    
                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_test, y_test, corpus_test)=Features.Extract_Features_Urls_Testing()
                # Tranform the list of dictionaries into a sparse matrix
                X_test=Features_Support.Vectorization_Testing(feature_list_dict_test, vectorizer)
                joblib.dump(X_test, os.path.join(url_test_dir, "X_test_unprocessed.pkl"))
                # TFIDF
                if config["HTML_Features"]["tfidf_websites"] == "True":
                    if flag_training==False:
                        tfidf_vectorizer=joblib.load( os.path.join(url_train_dir, "tfidf_vectorizer.pkl"))
                    logger.info("Extracting TFIDF features for testing websites ######")
                    Tfidf_test=Tfidf.tfidf_testing(corpus_test, tfidf_vectorizer)
                    joblib.dump(Tfidf_test, os.path.join(url_test_dir, "tfidf_features.pkl"))
                    X_test=hstack([X_test, Tfidf_test])
                
                joblib.dump(X_test, os.path.join(url_test_dir, "X_test_unprocessed_with_tfidf.pkl"))
                # Use Min_Max_scaling for prepocessing the feature matrix
                X_test=Features_Support.Preprocessing(X_test)
                joblib.dump(X_test, os.path.join(url_test_dir, "X_test_processed.pkl"))
                
                # Feature Selection
                if config["Feature Selection"]["select best features"]=="True":
                    if flag_training==False:
                        selection=joblib.load(os.path.join(url_train_dir, "selection.pkl"))
                    #k: Number of Best features
                    k = int(config["Feature Selection"]["number of best features"])
                    X_test = Feature_Selection.Select_Best_Features_Testing(X_test, selection, k, feature_list_dict_test)
                    joblib.dump(X_test, os.path.join(url_test_dir, "X_test_processed_best_features.pkl"))
                
                
                # Test on imbalanced datasets
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_test, y_test=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_test, y_test)
                #Dump Testing feature matrix with labels
                if not os.path.exists(url_test_dir):
                    os.makedirs(url_test_dir)
                joblib.dump(X_test, os.path.join(url_test_dir, "X_test.pkl"))
                joblib.dump(y_test, os.path.join(url_test_dir, "y_test.pkl"))
                logger.info("Feature Extraction for testing dataset: Done!")
            else:
                X_test = None
                y_test = None


    if config["Classification"]["Running the classifiers"]=="True":
        if Feature_extraction==False:
            if config["Classification"]["load model"] == "True":
                X_train, y_train, X_test, y_test, vectorizer_train, vectorizer_test = load_dataset(load_train=False, load_test=True)
                logger.info("loading test dataset only")
            else:
                X_train, y_train, X_test, y_test, vectorizer_train, vectorizer_test = load_dataset(load_train=True, load_test=True)
            if config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
                if config["Classification"]["load model"] == "False":
                    """
                    features_extracted=vectorizer_train.get_feature_names()
                    #logger.info(features_extracted)
                    import numpy as np
                    if X_train is not None:
                        Features_training=vectorizer_train.inverse_transform(X_train)
                    if X_test is not None:
                        Features_testing=vectorizer_test.inverse_transform(X_test)
                    mask=[]
                    #mask.append(0)
                    #logger.info("Section: {} ".format(section))
                    for feature in features_extracted:
                        feature_name=feature
                        if "=" in feature:
                            feature_name=feature.split("=")[0]
                        if "url_char_distance_" in feature:
                            feature_name="char_distance"
                        for section in ["HTML_Features", "URL_Features", "Network_Features", "Javascript_Features"]:
                            try:
                                if config[section][feature_name]=="True":
                                    if config[section][section.lower()]=="True":
                                        mask.append(1)
                                    else:
                                        mask.append(0)
                                else:
                                    mask.append(0)
                            except KeyError as e:
                                pass
                    logger.info(len(vectorizer_train.get_feature_names()))
                    vectorizer_train.restrict(mask)
                    url_classification_dir =  os.path.join(args.output_input_dir, "URLs_Classification")
                    if X_train is not None:
                        X_train=vectorizer_train.transform(Features_training)
                        logger.info(np.shape(X_train))
                    if X_test is not None:
                        X_test=vectorizer_train.transform(Features_testing)
                    if not os.path.exists(url_classification_dir):
                        os.makedirs(url_classification_dir)
                    joblib.dump(vectorizer_train, os.path.join(url_classification_dir, "vectorizer_restricted.pkl"))
                    if X_train is not None:
                        joblib.dump(X_train, os.path.join(url_classification_dir, "X_train_restricted.pkl"))
                    if X_test is not None:
                        joblib.dump(X_test, os.path.join(url_classification_dir, "X_test_restricted.pkl"))
                    logger.info(len(vectorizer_train.get_feature_names()))
                    """
                    #exit()
            elif config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
                if config["Classification"]["load model"] == "False":
                    features_extracted = vectorizer_train.get_feature_names()
                    logger.info(len(features_extracted))
                    mask= []
                    for feature_name in features_extracted:
                        if "=" in feature_name:
                            feature_name=feature_name.split("=")[0]
                        if "count_in_body" in feature_name:
                            if config["Email_Features"]["blacklisted_words_body"] == "True":
                                mask.append(1)
                            else:
                                mask.append(0)
                        elif "count_in_subject" in feature_name:
                            if config["Email_Features"]["blacklisted_words_subject"] == "True":
                                mask.append(1)
                            else:
                                mask.append(0)
                        else:
                            if config["Email_Features"][feature_name]=="True":
                                mask.append(1)
                            else:
                                mask.append(0)
                    logger.info(mask)
                    vectorizer=vectorizer_train.restrict(mask)
                    logger.info(len(vectorizer.get_feature_names()))
                #X_train=vectorizer.transform(X_train)

        logger.info("Running the Classifiers....")
        classifiers(X_train, y_train, X_test, y_test)
        logger.info("Done running the Classifiers!!")

if __name__ == "__main__":
    # execute only if run as a script
    answer = Confirmation()
    original = sys.stdout
    if answer is True:
        logger.debug("Running......")
        # sys.stdout= open("log.txt",'w')
        main()
        # sys.stdout=original
        logger.debug("Done!")
    sys.stdout=original
