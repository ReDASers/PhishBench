import os
import sys
from typing import List, Dict

import joblib
import pandas as pd
import phishbench.Feature_Selection as Feature_Selection
import phishbench.Features_Support as Features_Support
import phishbench.Tfidf as Tfidf
import phishbench.feature_extraction.legacy.email_features as legacy_email
import phishbench.feature_extraction.legacy.url_features as legacy_url
import phishbench.classification as classification
import phishbench.evaluation as evaluation
from phishbench.utils import Globals
from phishbench.utils import user_interaction
from scipy.sparse import hstack


def export_features_to_csv(features: List[Dict], y: List, file_path: str):
    """
    Exports raw features to csv
    Parameters
    ----------
    features : List[Dict]
        A list of dicts with each dict containing the features of a single data point
    y : List
        The labels for each datapoint. This should have the same length as features
    file_path : str
        The file to save the csv to
    """
    df = pd.DataFrame(features)
    df['is_phish'] = y
    df.to_csv(file_path, index=None)


def extract_url_train_features(url_train_dir: str, run_tfidf: bool):
    """
    Extracts features from the URL training dataset

    Parameters
    ----------
    url_train_dir : str
        The location of the url training dataset
    run_tfidf : bool
        Whether or not to run TF-IDF
    Returns
    -------
    X_train:
        A scipy sparse matrix containing the extracted features
    y_train:
        A list containing the labels for the extracted dataset
    vectorizer:
        The sklearn vectorizer for the features
    tfidf_vectorizer:
        The TF-IDF vectorizer used to generate TFIDF vectors. None if TF-IDF is not run
    """

    if not os.path.exists(url_train_dir):
        os.makedirs(url_train_dir)

    feature_list_dict_train, y_train, corpus_train = legacy_url.Extract_Features_Urls_Training()

    # Export features to csv
    if Globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(url_train_dir, 'features.csv')
        export_features_to_csv(feature_list_dict_train, y_train, out_path)

    # Transform the list of dictionaries into a sparse matrix
    X_train, vectorizer = Features_Support.Vectorization_Training(feature_list_dict_train)

    joblib.dump(X_train, os.path.join(url_train_dir, "X_train_unprocessed.pkl"))

    # Add tfidf if the user marked it as True
    if run_tfidf:
        Globals.logger.info("Extracting TFIDF features for training websites ###### ######")
        Tfidf_train, tfidf_vectorizer = Tfidf.tfidf_training(corpus_train)
        joblib.dump(Tfidf_train, os.path.join(url_train_dir, "tfidf_features.pkl"))
        X_train = hstack([X_train, Tfidf_train])
    else:
        tfidf_vectorizer = None

    X_train = Features_Support.Preprocessing(X_train)

    return X_train, y_train, vectorizer, tfidf_vectorizer


def extract_url_features_test(url_test_dir: str, vectorizer, tfidf_vectorizer=None):
    """
    Extracts features from the URL testing dataset
    Parameters
    ----------
    url_test_dir : str
        The folder containing the test url dataset
    vectorizer :
        The vectorizer used to vectorize the training dataset
    tfidf_vectorizer :
        The TF-IDF vectorizer used to extract TF-IDF vectors from the training datset. None if
        TF-IDF vectors were not extracted

    Returns
    -------
    x_test:
        A scipy sparse matrix containing the extracted features
    y_test
        A list containing the dataset labels
    """
    feature_list_dict_test, y_test, corpus_test = legacy_url.Extract_Features_Urls_Testing()

    if not os.path.exists(url_test_dir):
        os.makedirs(url_test_dir)

    # Export features to csv
    if Globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(url_test_dir, 'features.csv')
        export_features_to_csv(feature_list_dict_test, y_test, out_path)

    x_test = vectorizer.transform(feature_list_dict_test)

    joblib.dump(x_test, os.path.join(url_test_dir, "X_test_unprocessed.pkl"))

    if tfidf_vectorizer:
        Globals.logger.info("Extracting TFIDF features for testing websites ######")
        Tfidf_test = Tfidf.tfidf_testing(corpus_test, tfidf_vectorizer)
        joblib.dump(Tfidf_test, os.path.join(url_test_dir, "tfidf_features.pkl"))
        x_test = hstack([x_test, Tfidf_test])

    joblib.dump(x_test, os.path.join(url_test_dir, "X_test_unprocessed_with_tfidf.pkl"))

    # Use Min_Max_scaling for prepocessing the feature matrix
    x_test = Features_Support.Preprocessing(x_test)

    Globals.logger.info("Feature Extraction for testing dataset: Done!")

    return x_test, y_test


def extract_email_train_features(email_train_dir, run_tfidf):
    """
    Extracts features from the email training dataset

    Parameters
    ----------
    email_train_dir : str
        The location of the email training dataset
    run_tfidf : bool
        Whether or not to run TF-IDF
    Returns
    -------
    X_train:
        A scipy sparse matrix containing the extracted features
    y_train:
        A list containing the labels for the extracted dataset
    vectorizer:
        The sklearn vectorizer for the features
    tfidf_vectorizer:
        The TF-IDF vectorizer used to generate TFIDF vectors. None if TF-IDF is not run
    """
    if not os.path.exists(email_train_dir):
        os.makedirs(email_train_dir)
    print("Extracting Train Set")
    Globals.logger.info("Extracting Train Set")
    legit_path = Globals.config["Dataset Path"]["path_legitimate_training"]
    phish_path = Globals.config["Dataset Path"]["path_phishing_training"]

    feature_list_dict_train, y_train, corpus_train = legacy_email.extract_dataset_features(legit_path, phish_path)

    # Export features to csv
    if Globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(email_train_dir, 'features.csv')
        export_features_to_csv(feature_list_dict_train, y_train, out_path)

    # Tranform the list of dictionaries into a sparse matrix
    x_train, vectorizer = Features_Support.Vectorization_Training(feature_list_dict_train)

    joblib.dump(x_train, os.path.join(email_train_dir, "X_train_unprocessed.pkl"))

    if run_tfidf:
        Globals.logger.info("tfidf_emails_train ######")
        tfidf_train, tfidf_vectorizer = Tfidf.tfidf_training(corpus_train)
        joblib.dump(tfidf_train, os.path.join(email_train_dir, "tfidf_features.pkl"))
        x_train = hstack([x_train, tfidf_train])
    else:
        tfidf_vectorizer = None

    x_train = Features_Support.Preprocessing(x_train)

    return x_train, y_train, vectorizer, tfidf_vectorizer


def extract_email_test_features(email_test_dir, vectorizer=None, tfidf_vectorizer=None):
    """
    Extracts features from the email testing dataset
    Parameters
    ----------
    email_test_dir : str
        The folder containing the test email dataset
    vectorizer :
        The vectorizer used to vectorize the training dataset
    tfidf_vectorizer :
        The TF-IDF vectorizer used to extract TF-IDF vectors from the training datset. None if
        TF-IDF vectors were not extracted
    Returns
    -------
    x_test:
        A scipy sparse matrix containing the extracted features
    y_test
        A list containing the dataset labels
    """
    # Extract features in a dictionary for each email. return a list of dictionaries
    legit_path = Globals.config["Dataset Path"]["path_legitimate_testing"]
    phish_path = Globals.config["Dataset Path"]["path_phishing_testing"]
    print("Extracting Test Set")
    Globals.loger.info('Extracting Test Set')
    feature_list_dict_test, y_test, corpus_test = legacy_email.extract_dataset_features(legit_path, phish_path)

    # Export features to csv
    if Globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(email_test_dir, 'features.csv')
        export_features_to_csv(feature_list_dict_test, y_test, out_path)

    # Tranform the list of dictionaries into a sparse matrix
    x_test = vectorizer.transform(feature_list_dict_test)

    # Add tfidf if the user marked it as True
    if tfidf_vectorizer:
        Globals.logger.info("tfidf_emails_train ######")
        tfidf_test = Tfidf.tfidf_testing(corpus_test, tfidf_vectorizer)
        x_test = hstack([x_test, tfidf_test])

    # Use Min_Max_scaling for prepocessing the feature matrix
    x_test = Features_Support.Preprocessing(x_test)

    # Dump Testing feature matrix with labels
    if not os.path.exists(email_test_dir):
        os.makedirs(email_test_dir)

    Globals.logger.info("Feature Extraction for testing dataset: Done!")

    return x_test, y_test


def extract_email_features():
    """
    Extracts features from a email dataset. If PhishBench is configured to only extract features from a test dataset,
    this function will automatically load pre-extracted training data from disk.
    Returns
    -------
    X_train:
        A scipy sparse matrix containing the extracted features
    y_train:
        A list containing the labels for the extracted dataset
    x_test:
        A scipy sparse matrix containing the extracted features
    y_test
        A list containing the dataset labels
    vectorizer:
        The sklearn vectorizer for the features
    tfidf_vectorizer:
        The TF-IDF vectorizer used to generate TFIDF vectors. None if TF-IDF is not run
    """
    email_train_dir = os.path.join(Globals.args.output_input_dir, "Emails_Training")
    email_test_dir = os.path.join(Globals.args.output_input_dir, "Emails_Testing")
    run_tfidf = Globals.config['Email_Features'].getboolean('extract body features') \
                and Globals.config["Email_Body_Features"].getboolean("tfidf_emails")

    if Globals.config["Extraction"].getboolean("Training Dataset"):
        x_train, y_train, vectorizer, tfidf_vectorizer = extract_email_train_features(email_train_dir, run_tfidf)

        # Save features for training dataset
        joblib.dump(x_train, os.path.join(email_train_dir, "X_train.pkl"))
        joblib.dump(y_train, os.path.join(email_train_dir, "y_train.pkl"))
        joblib.dump(vectorizer, os.path.join(email_train_dir, "vectorizer.pkl"))
        if tfidf_vectorizer:
            joblib.dump(tfidf_vectorizer, os.path.join(email_train_dir, "tfidf_vectorizer.pkl"))
        Globals.logger.info("Feature Extraction for training dataset: Done!")
    else:
        x_train = joblib.load(os.path.join(email_train_dir, "X_train.pkl"))
        y_train = joblib.load(os.path.join(email_train_dir, "y_train.pkl"))
        vectorizer = joblib.load(os.path.join(email_train_dir, "vectorizer.pkl"))
        if run_tfidf:
            tfidf_vectorizer = joblib.load(os.path.join(email_train_dir, "tfidf_vectorizer.pkl"))

    if Globals.config["Extraction"]["Testing Dataset"] == "True":
        x_test, y_test = extract_email_test_features(email_test_dir, vectorizer, tfidf_vectorizer)
        joblib.dump(x_test, os.path.join(email_test_dir, "X_test.pkl"))
        joblib.dump(y_test, os.path.join(email_test_dir, "y_test.pkl"))
    else:
        x_test = None
        y_test = None
    return x_train, y_train, x_test, y_test, vectorizer, tfidf_vectorizer


def extract_url_features():
    """
    Extracts features from a URL dataset. If PhishBench is configured to only extract features from a test dataset,
    this function will automatically load pre-extracted training data from disk.
    Returns
    -------
    X_train:
        A scipy sparse matrix containing the extracted features
    y_train:
        A list containing the labels for the extracted dataset
    x_test:
        A scipy sparse matrix containing the extracted features
    y_test
        A list containing the dataset labels
    vectorizer:
        The sklearn vectorizer for the features
    tfidf_vectorizer:
        The TF-IDF vectorizer used to generate TFIDF vectors. None if TF-IDF is not run
    """
    url_train_dir = os.path.join(Globals.args.output_input_dir, "URLs_Training")
    url_test_dir = os.path.join(Globals.args.output_input_dir, "URLs_Testing")
    run_tfidf = Globals.config["URL_Feature_Types"].getboolean("HTML") and \
                Globals.config["HTML_Features"].getboolean("tfidf_websites")

    if Globals.config["Extraction"].getboolean("Training Dataset"):
        x_train, y_train, vectorizer, tfidf_vectorizer = extract_url_train_features(url_train_dir, run_tfidf)

        # dump features and labels and vectorizers
        joblib.dump(x_train, os.path.join(url_train_dir, "X_train.pkl"))
        joblib.dump(y_train, os.path.join(url_train_dir, "y_train.pkl"))
        joblib.dump(vectorizer, os.path.join(url_train_dir, "vectorizer.pkl"))
        if tfidf_vectorizer:
            joblib.dump(tfidf_vectorizer, os.path.join(url_train_dir, "tfidf_vectorizer.pkl"))
        Globals.logger.info("Feature Extraction for training dataset: Done!")
    else:
        # if training was done in another instance of the platform then load the necessary files
        x_train = joblib.load(os.path.join(url_train_dir, "X_train.pkl"))
        y_train = joblib.load(os.path.join(url_train_dir, "y_train.pkl"))
        vectorizer = joblib.load(os.path.join(url_train_dir, "vectorizer.pkl"))
        # TFIDF
        if run_tfidf:
            tfidf_vectorizer = joblib.load(os.path.join(url_train_dir, "tfidf_vectorizer.pkl"))
        else:
            tfidf_vectorizer = None

    if Globals.config["Extraction"].getboolean("Testing Dataset"):

        x_test, y_test = extract_url_features_test(url_test_dir, vectorizer, tfidf_vectorizer)

        joblib.dump(x_test, os.path.join(url_test_dir, "X_test.pkl"))
        joblib.dump(y_test, os.path.join(url_test_dir, "y_test.pkl"))
        Globals.logger.info("Feature Extraction for testing dataset: Done!")
    else:
        x_test = None
        y_test = None
    return x_train, y_train, x_test, y_test, vectorizer, tfidf_vectorizer


def create_performance_df(scores: List[Dict]):
    df = pd.DataFrame(scores)
    columns: List = df.columns.tolist()
    columns.remove("classifier")
    columns.insert(0, "classifier")
    return df.reindex(columns=columns)


def run_phishbench():
    feature_extraction_flag = False  # flag for feature extraction

    if Globals.config["Extraction"]["Feature Extraction"] == 'True':
        feature_extraction_flag = True
        if Globals.config["Email or URL feature Extraction"].getboolean("extract_features_emails"):
            x_train, y_train, x_test, y_test, vectorizer, tfidf_vectorizer = extract_email_features()
        elif Globals.config["Email or URL feature Extraction"].getboolean("extract_features_urls"):
            x_train, y_train, x_test, y_test, vectorizer, tfidf_vectorizer = extract_url_features()
    else:
        if Globals.config["Email or URL feature Extraction"].getboolean("extract_features_emails"):
            train_dir = os.path.join(Globals.args.output_input_dir, "Emails_Training")
            test_dir = os.path.join(Globals.args.output_input_dir, "Emails_Testing")
            run_tfidf = Globals.config['Email_Features'].getboolean('extract body features') and \
                Globals.config["Email_Body_Features"].getboolean("tfidf_emails")
        elif Globals.config["Email or URL feature Extraction"].getboolean("extract_features_urls"):
            train_dir = os.path.join(Globals.args.output_input_dir, "URLs_Training")
            test_dir = os.path.join(Globals.args.output_input_dir, "URLs_Testing")
            run_tfidf = Globals.config["URL_Feature_Types"].getboolean("HTML") and \
                Globals.config["HTML_Features"].getboolean("tfidf_websites")
        x_train = joblib.load(os.path.join(train_dir, "X_train.pkl"))
        y_train = joblib.load(os.path.join(train_dir, "y_train.pkl"))
        vectorizer = joblib.load(os.path.join(train_dir, "vectorizer.pkl"))
        if run_tfidf:
            tfidf_vectorizer = joblib.load(os.path.join(train_dir, "tfidf_vectorizer.pkl"))
        else:
            tfidf_vectorizer = None
        if os.path.exists(os.path.join(test_dir, "X_test.pkl")):
            x_test = joblib.load(os.path.join(test_dir, "X_test.pkl"))
            y_test = joblib.load(os.path.join(test_dir, "y_test.pkl"))
        else:
            x_test = None
            y_test = None

    # Feature Selection
    if Globals.config["Feature Selection"].getboolean("select best features"):
        ranking_dir = os.path.join(Globals.args.output_input_dir, "Feature_Ranking")
        if not os.path.exists(ranking_dir):
            os.makedirs(ranking_dir)
        # k: Number of Best features
        num_best_features = int(Globals.config["Feature Selection"]["number of best features"])
        x_train, selection_model = Feature_Selection.Feature_Ranking(x_train, y_train, num_best_features,
                                                                     vectorizer, tfidf_vectorizer)
        # Dump model
        joblib.dump(selection_model, os.path.join(ranking_dir, "selection.pkl"))
        joblib.dump(x_train, os.path.join(ranking_dir, "X_train_processed_best_features.pkl"))

        if x_test is not None:
            x_test = selection_model.transform(x_test)
            Globals.logger.info("X_test Shape: {}".format(x_test.shape))
            joblib.dump(x_test, os.path.join(ranking_dir, "X_test_processed_best_features.pkl"))

    if classification.settings.run_classifiers():
        folder = os.path.join(Globals.args.output_input_dir, "Classifiers")
        print("Training Classifiers")

        classifiers = classification.train_classifiers(x_train, y_train, io_dir=folder)
        performance_list_dict = []
        for classifier in classifiers:
            metrics = evaluation.evaluate_classifier(classifier, x_test, y_test)
            metrics['classifier'] = classifier.name
            performance_list_dict.append(metrics)
        classifier_performances = create_performance_df(performance_list_dict)

        print(classifier_performances)
        classifier_performances.to_csv(os.path.join(folder, "performance.csv"), index=False)

def main():
    # execute only if run as a script
    Globals.setup_globals()
    answer = user_interaction.Confirmation(Globals.args.ignore_confirmation)
    original = sys.stdout
    if answer:
        Globals.logger.debug("Running......")
        run_phishbench()
        Globals.logger.debug("Done!")
    sys.stdout = original
    Globals.destroy_globals()


if __name__ == "__main__":
    main()
