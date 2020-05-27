import os
import sys

import joblib
import pandas as pd
from scipy.sparse import hstack

import phishbench.Feature_Selection as Feature_Selection
import phishbench.Features_Support as Features_Support
import phishbench.Tfidf as Tfidf
import phishbench.feature_extraction.legacy.email_features as legacy_email
import phishbench.feature_extraction.legacy.url_features as legacy_url
from phishbench.Classifiers import classifiers
from phishbench.dataset import dataset
from phishbench.utils import Globals
from phishbench.utils import user_interaction


def export_features_to_csv(feature_list_dict,y,loc):
    df = pd.DataFrame(feature_list_dict)
    df['is_phish'] = y
    df.to_csv(loc, index=None)


def extract_url_features_test(url_train_dir, url_test_dir, vectorizer, tfidf_vectorizer=None,
                              feature_selection_model=None):
    # Extract features in a dictionnary for each email. return a list of dictionaries
    feature_list_dict_test, y_test, corpus_test = legacy_url.Extract_Features_Urls_Testing()

    if not os.path.exists(url_test_dir):
        os.makedirs(url_test_dir)

    # Export features to csv
    if Globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(url_test_dir, 'features.csv')
        export_features_to_csv(feature_list_dict_test, y_test, out_path)

    # Tranform the list of dictionaries into a sparse matrix
    X_test = vectorizer.transform(feature_list_dict_test)

    joblib.dump(X_test, os.path.join(url_test_dir, "X_test_unprocessed.pkl"))

    if tfidf_vectorizer:
        Globals.logger.info("Extracting TFIDF features for testing websites ######")
        Tfidf_test = Tfidf.tfidf_testing(corpus_test, tfidf_vectorizer)
        joblib.dump(Tfidf_test, os.path.join(url_test_dir, "tfidf_features.pkl"))
        X_test = hstack([X_test, Tfidf_test])

    joblib.dump(X_test, os.path.join(url_test_dir, "X_test_unprocessed_with_tfidf.pkl"))

    # Use Min_Max_scaling for prepocessing the feature matrix
    X_test = Features_Support.Preprocessing(X_test)
    joblib.dump(X_test, os.path.join(url_test_dir, "X_test_processed.pkl"))

    # Feature Selection
    if Globals.config["Feature Selection"]["select best features"] == "True":
        if not feature_selection_model:
            feature_selection_model = joblib.load(os.path.join(url_train_dir, "selection.pkl"))
        # k: Number of Best features
        k = int(Globals.config["Feature Selection"]["number of best features"])
        X_test = feature_selection_model.transform(X_test)
        Globals.logger.info("X_test Shape: {}".format(X_test.shape))
        joblib.dump(X_test, os.path.join(url_test_dir, "X_test_processed_best_features.pkl"))

    # Dump Testing feature matrix with labels
    joblib.dump(X_test, os.path.join(url_test_dir, "X_test.pkl"))
    joblib.dump(y_test, os.path.join(url_test_dir, "y_test.pkl"))
    Globals.logger.info("Feature Extraction for testing dataset: Done!")

    return X_test, y_test


def extract_email_features_test(email_train_dir, email_test_dir, vectorizer=None, tfidf_vectorizer=None,
                                selection=None):
    if not vectorizer:
        X_train = joblib.load(os.path.join(email_train_dir, "X_train.pkl"))
        y_train = joblib.load(os.path.join(email_train_dir, "y_train.pkl"))
        vectorizer = joblib.load(os.path.join(email_train_dir, "vectorizer.pkl"))

    # Extract features in a dictionnary for each email. return a list of dictionaries
    feature_list_dict_test, y_test, corpus_test = legacy_email.Extract_Features_Emails_Testing()

    # Export features to csv
    if Globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(email_test_dir, 'features.csv')
        export_features_to_csv(feature_list_dict_test, y_test, out_path)

    # Tranform the list of dictionaries into a sparse matrix
    X_test = vectorizer.transform(feature_list_dict_test)

    # Add tfidf if the user marked it as True
    if Globals.config["Email_Body_Features"]["tfidf_emails"] == "True":
        if not tfidf_vectorizer:
            tfidf_vectorizer = joblib.load(os.path.join(email_train_dir, "tfidf_vectorizer.pkl"))
        Globals.logger.info("tfidf_emails_train ######")
        Tfidf_test = Tfidf.tfidf_testing(corpus_test, tfidf_vectorizer)
        X_test = hstack([X_test, Tfidf_test])

    # Use Min_Max_scaling for prepocessing the feature matrix
    X_test = Features_Support.Preprocessing(X_test)
    # feature ranking
    if Globals.config["Feature Selection"]["select best features"] == "True":
        if not selection:
            selection = joblib.load(os.path.join(email_train_dir, "selection.pkl"))
        # k: Number of Best features
        k = int(Globals.config["Feature Selection"]["number of best features"])
        X_test = selection.transform(X_test)
        Globals.logger.info("X_Shape: {}".format(X_test.shape))
        Globals.logger.info("### Feature Ranking and Selection for Training Done!")

    # Dump Testing feature matrix with labels
    if not os.path.exists(email_test_dir):
        os.makedirs(email_test_dir)
    joblib.dump(X_test, os.path.join(email_test_dir, "X_test.pkl"))
    joblib.dump(y_test, os.path.join(email_test_dir, "y_test.pkl"))

    Globals.logger.info("Feature Extraction for testing dataset: Done!")

    if X_train:
        return X_train, y_train, vectorizer, X_test, y_test

    return X_test, y_test

def extract_email_train_features(email_train_dir):
    if not os.path.exists(email_train_dir):
        os.makedirs(email_train_dir)
    print("Extracting Training Set")

    feature_list_dict_train, y_train, corpus_train = legacy_email.Extract_Features_Emails_Training()

    # Export features to csv
    if Globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(email_train_dir, 'features.csv')
        export_features_to_csv(feature_list_dict_train, y_train, out_path)

    # Tranform the list of dictionaries into a sparse matrix
    X_train, vectorizer = Features_Support.Vectorization_Training(feature_list_dict_train)
    # Save model for vectorization
    joblib.dump(vectorizer, os.path.join(email_train_dir, "vectorizer.pkl"))
    joblib.dump(X_train, os.path.join(email_train_dir, "X_train_unprocessed.pkl"))

    # Add tfidf if the user marked it as True
    if Globals.config["Email_Body_Features"].getboolean("tfidf_emails"):
        Globals.logger.info("tfidf_emails_train ######")
        Tfidf_train, tfidf_vectorizer = Tfidf.tfidf_training(corpus_train)
        joblib.dump(Tfidf_train, os.path.join(email_train_dir, "tfidf_features.pkl"))
        X_train = hstack([X_train, Tfidf_train])
        # Save tfidf vectorizer
        joblib.dump(tfidf_vectorizer, os.path.join(email_train_dir, "tfidf_vectorizer.pkl"))
    else:
        tfidf_vectorizer = None

    X_train = Features_Support.Preprocessing(X_train)
    joblib.dump(X_train, os.path.join(email_train_dir, "X_train_processed.pkl"))
    return X_train, y_train, vectorizer, tfidf_vectorizer


def extract_url_train_features(url_train_dir):
    # Create directory to store dada
    if not os.path.exists(url_train_dir):
        os.makedirs(url_train_dir)

    feature_list_dict_train, y_train, corpus_train = legacy_url.Extract_Features_Urls_Training()

    # Export features to csv
    if Globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(url_train_dir, 'features.csv')
        export_features_to_csv(feature_list_dict_train, y_train, out_path)

    # Tranform the list of dictionaries into a sparse matrix
    X_train, vectorizer = Features_Support.Vectorization_Training(feature_list_dict_train)

    # Dump vectorizer
    joblib.dump(vectorizer, os.path.join(url_train_dir, "vectorizer.pkl"))
    joblib.dump(X_train, os.path.join(url_train_dir, "X_train_unprocessed.pkl"))

    # Add tfidf if the user marked it as True
    if Globals.config["URL_Feature_Types"].getboolean("HTML") and \
            Globals.config["HTML_Features"].getboolean("tfidf_websites"):
        Globals.logger.info("Extracting TFIDF features for training websites ###### ######")
        Tfidf_train, tfidf_vectorizer = Tfidf.tfidf_training(corpus_train)
        joblib.dump(Tfidf_train, os.path.join(url_train_dir, "tfidf_features.pkl"))
        X_train = hstack([X_train, Tfidf_train])
        # dump tfidf vectorizer
        joblib.dump(tfidf_vectorizer, os.path.join(url_train_dir, "tfidf_vectorizer.pkl"))
    else:
        tfidf_vectorizer = None

    X_train = Features_Support.Preprocessing(X_train)
    joblib.dump(X_train, os.path.join(url_train_dir, "X_train_processed.pkl"))
    return X_train, y_train, vectorizer, tfidf_vectorizer


def extract_train_features(email_train_dir, url_train_dir):
    if Globals.config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
        return extract_email_train_features(email_train_dir)
    elif Globals.config["Email or URL feature Extraction"]["extract_features_URLs"] == "True":
        return extract_url_train_features(url_train_dir)
    return None, None, None, None


def run_phishbench():
    feature_extraction_flag = False  # flag for feature extraction
    flag_training = False
    # Feature dumping and loading methods
    # flag_saving_pickle=Globals.config["Features Format"]["Pikle"]
    # flag_saving_svmlight=Globals.config["Features Format"]["Svmlight format"]

    # Create Data Dump directory if doesn't exist
    ranking_dir = os.path.join(Globals.args.output_input_dir, "Feature_Ranking")
    if not os.path.exists(ranking_dir):
        os.makedirs(ranking_dir)
    email_train_dir = os.path.join(Globals.args.output_input_dir, "Emails_Training")
    email_test_dir = os.path.join(Globals.args.output_input_dir, "Emails_Testing")
    url_train_dir = os.path.join(Globals.args.output_input_dir, "URLs_Training")
    url_test_dir = os.path.join(Globals.args.output_input_dir, "URLs_Testing")
    ######################################################
    ############## Feature ranking only ##################
    ######################################################
    if Globals.config["Feature Selection"]["Feature Ranking Only"] == 'True':
        if Globals.config["Extraction"]["feature extraction"] == "True":
            X, y, vectorizer, tfidf_vectorizer = extract_train_features(email_train_dir, url_train_dir)
        else:
            X, y, X_test, y_test, vectorizer_train, vectorizer_test = dataset.load_dataset()
            # feature_list_dict_train=vectorizer_train.inverse_transform(X)

        Globals.logger.info("Select Best Features ######")
        num_best_features = int(Globals.config["Feature Selection"]["number of best features"])
        # X, selection = Feature_Selection.Select_Best_Features_Training(X, y, k)
        X, selection_model = Feature_Selection.Feature_Ranking(X, y, num_best_features)
        if Globals.config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
            joblib.dump(selection_model, os.path.join(email_train_dir, "selection.pkl"))
        elif Globals.config["Email or URL feature Extraction"]["extract_features_URLs"] == "True":
            joblib.dump(selection_model, os.path.join(url_train_dir, "selection.pkl"))

    ######################################################
    ################ Feature Extraction ##################
    ######################################################
    elif Globals.config["Extraction"]["Feature Extraction"] == 'True':
        feature_extraction_flag = True
        ################ Email Feature Extraction ##################
        if Globals.config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
            if Globals.config["Extraction"]["Training Dataset"] == "True":
                X_train, y_train, vectorizer, tfidf_vectorizer = extract_email_train_features(email_train_dir)

                # feature ranking
                if Globals.config["Feature Selection"]["select best features"] == "True":
                    # k: Number of Best features
                    Globals.logger.info("Select Best Features ######")
                    num_best_features = int(Globals.config["Feature Selection"]["number of best features"])
                    # X_train, selection = Feature_Selection.Select_Best_Features_Training(X_train, y_train, k)
                    X_train, selection_model = Feature_Selection.Feature_Ranking(X_train, y_train, num_best_features)
                    # dump selection model
                    joblib.dump(selection_model, os.path.join(email_train_dir, "selection.pkl"))
                    Globals.logger.info("### Feature Ranking and Selection for Training Done!")

                # Save features for training dataset
                joblib.dump(X_train, os.path.join(email_train_dir, "X_train.pkl"))
                joblib.dump(y_train, os.path.join(email_train_dir, "y_train.pkl"))

                # flag to mark if training was done
                flag_training = True
                Globals.logger.info("Feature Extraction for training dataset: Done!")

            if Globals.config["Extraction"]["Testing Dataset"] == "True":
                X_test, y_test = extract_email_features_test(email_train_dir, email_test_dir)

            else:
                X_test = None
                y_test = None

        ################ URL Feature Extraction ##################
        elif Globals.config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
            if Globals.config["Extraction"]["Training Dataset"] == "True":
                # Extract features in a dictionnary for each url. return a list of dictionaries
                feature_list_dict_train, y_train, corpus_train = legacy_url.Extract_Features_Urls_Training()
                X_train, y_train, vectorizer, tfidf_vectorizer = extract_url_train_features(url_train_dir)

                # Feature Selection
                if Globals.config["Feature Selection"]["select best features"] == "True":
                    # k: Number of Best features
                    num_best_features = int(Globals.config["Feature Selection"]["number of best features"])
                    X_train, selection_model = Feature_Selection.Feature_Ranking(X_train, y_train, num_best_features)
                    # Dump model
                    joblib.dump(selection_model, os.path.join(url_train_dir, "selection.pkl"))
                    joblib.dump(X_train, os.path.join(url_train_dir, "X_train_processed_best_features.pkl"))

                # dump features and labels and vectorizers
                joblib.dump(X_train, os.path.join(url_train_dir, "X_train.pkl"))
                joblib.dump(y_train, os.path.join(url_train_dir, "y_train.pkl"))
                if Globals.config["Features Export"]["csv"] == "True":
                    export_filename = os.path.join(url_train_dir, "Features.csv")
                    df = pd.DataFrame(feature_list_dict_train)
                    df["is_phish"] = y_train
                    df.to_csv(export_filename, index=None)
                # flag to mark if training was done
                flag_training = True
                Globals.logger.info("Feature Extraction for training dataset: Done!")

            if Globals.config["Extraction"]["Testing Dataset"] == "True":
                if not flag_training:
                    # if training was done in another instance of the plaform then load the necessary files
                    X_train = joblib.load(os.path.join(url_train_dir, "X_train.pkl"))
                    y_train = joblib.load(os.path.join(url_train_dir, "y_train.pkl"))
                    vectorizer = joblib.load(os.path.join(url_train_dir, "vectorizer.pkl"))
                    # TFIDF
                    run_tfidf = Globals.config["URL_Feature_Types"].getboolean("HTML") and \
                                Globals.config["HTML_Features"].getboolean("tfidf_websites")
                    if run_tfidf:
                        tfidf_vectorizer = joblib.load(os.path.join(url_train_dir, "tfidf_vectorizer.pkl"))
                    else:
                        tfidf_vectorizer = None
                X_test, y_test = extract_url_features_test(url_train_dir, url_test_dir, vectorizer, tfidf_vectorizer)

                # # Feature Selection
                # if Globals.config["Feature Selection"]["select best features"] == "True":
                #     if flag_training == False:
                #         selection_model = joblib.load(os.path.join(url_train_dir, "selection.pkl"))
                #     # k: Number of Best features
                #     num_best_features = int(Globals.config["Feature Selection"]["number of best features"])
                #     print(X_test.shape)
                #     X_test = selection_model.transform(X_test)
                #     joblib.dump(X_test, os.path.join(url_test_dir, "X_test_processed_best_features.pkl"))

                # Dump Testing feature matrix with labels
                if not os.path.exists(url_test_dir):
                    os.makedirs(url_test_dir)
                joblib.dump(X_test, os.path.join(url_test_dir, "X_test.pkl"))
                joblib.dump(y_test, os.path.join(url_test_dir, "y_test.pkl"))
                Globals.logger.info("Feature Extraction for testing dataset: Done!")
            else:
                X_test = None
                y_test = None

    if Globals.config["Classification"]["Running the classifiers"] == "True":
        if not feature_extraction_flag:
            if Globals.config["Classification"]["load model"] == "True":
                X_train, y_train, X_test, y_test, vectorizer_train, vectorizer_test = dataset.load_dataset(
                    load_train=False, load_test=True)
                Globals.logger.info("loading test dataset only")
            else:
                X_train, y_train, X_test, y_test, vectorizer_train, vectorizer_test = dataset.load_dataset(
                    load_train=True, load_test=True)
            if Globals.config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
                if Globals.config["Classification"]["load model"] == "False":
                    pass
                    #
                    # features_extracted=vectorizer_train.get_feature_names()
                    # #Globals.logger.info(features_extracted)
                    # import numpy as np
                    # if X_train is not None:
                    #     Features_training=vectorizer_train.inverse_transform(X_train)
                    # if X_test is not None:
                    #     Features_testing=vectorizer_test.inverse_transform(X_test)
                    # mask=[]
                    # #mask.append(0)
                    # #Globals.logger.info("Section: {} ".format(section))
                    # for feature in features_extracted:
                    #     feature_name=feature
                    #     if "=" in feature:
                    #         feature_name=feature.split("=")[0]
                    #     if "url_char_distance_" in feature:
                    #         feature_name="char_distance"
                    #     for section in ["HTML_Features", "URL_Features", "Network_Features", "Javascript_Features"]:
                    #         try:
                    #             if Globals.config[section][feature_name]=="True":
                    #                 if Globals.config[section][section.lower()]=="True":
                    #                     mask.append(1)
                    #                 else:
                    #                     mask.append(0)
                    #             else:
                    #                 mask.append(0)
                    #         except KeyError as e:
                    #             pass
                    # Globals.logger.info(len(vectorizer_train.get_feature_names()))
                    # vectorizer_train.restrict(mask)
                    # url_classification_dir =  os.path.join(Globals.args.output_input_dir, "URLs_Classification")
                    # if X_train is not None:
                    #     X_train=vectorizer_train.transform(Features_training)
                    #     Globals.logger.info(np.shape(X_train))
                    # if X_test is not None:
                    #     X_test=vectorizer_train.transform(Features_testing)
                    # if not os.path.exists(url_classification_dir):
                    #     os.makedirs(url_classification_dir)
                    # joblib.dump(vectorizer_train, os.path.join(url_classification_dir, "vectorizer_restricted.pkl"))
                    # if X_train is not None:
                    #     joblib.dump(X_train, os.path.join(url_classification_dir, "X_train_restricted.pkl"))
                    # if X_test is not None:
                    #     joblib.dump(X_test, os.path.join(url_classification_dir, "X_test_restricted.pkl"))
                    # Globals.logger.info(len(vectorizer_train.get_feature_names()))

                    # exit()
            elif Globals.config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
                if Globals.config["Classification"]["load model"] == "False":
                    features_extracted = vectorizer_train.get_feature_names()
                    Globals.logger.info(len(features_extracted))
                    mask = []
                    for feature_name in features_extracted:
                        if "=" in feature_name:
                            feature_name = feature_name.split("=")[0]
                        if "count_in_body" in feature_name:
                            if Globals.config["Email_Features"]["blacklisted_words_body"] == "True":
                                mask.append(1)
                            else:
                                mask.append(0)
                        elif "count_in_subject" in feature_name:
                            if Globals.config["Email_Features"]["blacklisted_words_subject"] == "True":
                                mask.append(1)
                            else:
                                mask.append(0)
                        else:
                            if Globals.config["Email_Features"][feature_name] == "True":
                                mask.append(1)
                            else:
                                mask.append(0)
                    Globals.logger.info(mask)
                    vectorizer = vectorizer_train.restrict(mask)
                    Globals.logger.info(len(vectorizer.get_feature_names()))
                # X_train=vectorizer.transform(X_train)

        Globals.logger.info("Running the Classifiers....")
        classifiers(X_train, y_train, X_test, y_test)
        Globals.logger.info("Done running the Classifiers!!")


def main():
    # execute only if run as a script
    Globals.setup_globals()
    answer = user_interaction.Confirmation(Globals.args.ignore_confirmation)
    original = sys.stdout
    if answer is True:
        Globals.logger.debug("Running......")
        # sys.stdout= open("log.txt",'w')
        run_phishbench()
        # sys.stdout=original
        Globals.logger.debug("Done!")
    sys.stdout = original
    Globals.destroy_globals()


if __name__ == "__main__":
    main()
