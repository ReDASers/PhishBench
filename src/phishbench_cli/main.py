import os
import sys
from typing import List, Dict

import joblib
import pandas as pd

import phishbench
import phishbench.Feature_Selection as Feature_Selection
import phishbench.Features_Support as Features_Support
import phishbench.classification as classification
import phishbench.evaluation as evaluation
import phishbench.feature_extraction.email.features
import phishbench.feature_extraction.email as email_extraction
import phishbench.feature_extraction.url as url_extraction
import phishbench.feature_preprocessing as preprocessing
import phishbench.input as pb_input
import phishbench.settings
from phishbench.feature_extraction import settings as extraction_settings
from phishbench.feature_extraction.reflection import FeatureClass, FeatureType
from phishbench.utils import phishbench_globals
from phishbench_cli import user_interaction


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
    df.to_csv(file_path, index=False)


def extract_url_train_features(output_dir: str, features: List[FeatureClass]):
    """
    Extracts features from the URL training dataset

    Parameters
    ----------
    output_dir : str
        The folder to output pickles
    features:
        The features to extract
    Returns
    -------
    X_train:
        A scipy sparse matrix containing the extracted features
    y_train:
        A list containing the labels for the extracted dataset
    vectorizer: preprocessing.Vectorizer
        The `Vectorizer` for the features
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    urls, labels = pb_input.read_train_set(extraction_settings.download_url_flag())

    print("Extracting Features")
    for feature in features:
        feature.fit(urls, labels)
    feature_list_dict = url_extraction.extract_features_from_list_urls(urls, features)
    print("Cleaning features")
    preprocessing.clean_features(feature_list_dict)

    # Export features to csv
    if phishbench_globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(output_dir, 'features.csv')
        export_features_to_csv(feature_list_dict, labels, out_path)

    # Transform the list of dictionaries into a sparse matrix
    vectorizer = preprocessing.Vectorizer()
    x_train = vectorizer.fit_transform(feature_list_dict)

    joblib.dump(x_train, os.path.join(output_dir, "X_train_unprocessed.pkl"))

    x_train = Features_Support.Preprocessing(x_train)

    return x_train, labels, vectorizer


def extract_url_features_test(output_dir: str, features: List[FeatureClass], vectorizer: preprocessing.Vectorizer):
    """
    Extracts features from the URL testing dataset

    Parameters
    ----------
    output_dir : str
        The folder to output pickles
    features:
        The features to extract
    vectorizer :
        The vectorizer used to vectorize the training dataset

    Returns
    -------
    x_test:
        A scipy sparse matrix containing the extracted features
    y_test
        A list containing the dataset labels
    """

    urls, labels = pb_input.read_test_set(extraction_settings.download_url_flag())

    print("Extracting Features")
    feature_list_dict = url_extraction.extract_features_from_list_urls(urls, features)
    print("Cleaning features")
    preprocessing.clean_features(feature_list_dict)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export features to csv
    if phishbench_globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(output_dir, 'features.csv')
        export_features_to_csv(feature_list_dict, labels, out_path)

    x_test = vectorizer.transform(feature_list_dict)

    joblib.dump(x_test, os.path.join(output_dir, "X_test_unprocessed.pkl"))

    # Use Min_Max_scaling for prepocessing the feature matrix
    x_test = Features_Support.Preprocessing(x_test)

    phishbench_globals.logger.info("Feature Extraction for testing dataset: Done!")

    return x_test, labels


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
    url_train_dir = os.path.join(phishbench_globals.args.output_input_dir, "URLs_Training")
    url_test_dir = os.path.join(phishbench_globals.args.output_input_dir, "URLs_Testing")

    features = url_extraction.create_new_features()

    if phishbench_globals.config["Extraction"].getboolean("Training Dataset"):
        x_train, y_train, vectorizer = extract_url_train_features(url_train_dir, features)

        # dump features and labels and vectorizers
        joblib.dump(x_train, os.path.join(url_train_dir, "X_train.pkl"))
        joblib.dump(y_train, os.path.join(url_train_dir, "y_train.pkl"))
        joblib.dump(vectorizer, os.path.join(url_train_dir, "vectorizer.pkl"))
        if not os.path.isdir(os.path.join(url_train_dir, "features")):
            os.makedirs(os.path.join(url_train_dir, "features"))
        for feature in features:
            path = os.path.join(url_train_dir, "features", f"{feature.config_name}.pkl")
            feature.save_state(path)
        phishbench_globals.logger.info("Feature Extraction for training dataset: Done!")
    else:
        # if training was done in another instance of the platform then load the necessary files
        x_train = joblib.load(os.path.join(url_train_dir, "X_train.pkl"))
        y_train = joblib.load(os.path.join(url_train_dir, "y_train.pkl"))
        vectorizer = joblib.load(os.path.join(url_train_dir, "vectorizer.pkl"))
        for feature in features:
            path = os.path.join(url_train_dir, "features", f"{feature.config_name}.pkl")
            feature.load_state(path)

    if phishbench_globals.config["Extraction"].getboolean("Testing Dataset"):
        x_test, y_test = extract_url_features_test(url_test_dir, features, vectorizer)

        joblib.dump(x_test, os.path.join(url_test_dir, "X_test.pkl"))
        joblib.dump(y_test, os.path.join(url_test_dir, "y_test.pkl"))
        phishbench_globals.logger.info("Feature Extraction for testing dataset: Done!")
    else:
        x_test = None
        y_test = None

    tfidf_vectorizer = None
    for feature in features:
        if isinstance(feature, url_extraction.url_features.internal_features.WebsiteTfidf):
            tfidf_vectorizer = feature.tfidf_vectorizer

    return x_train, y_train, x_test, y_test, vectorizer.scalar_vectorizer, tfidf_vectorizer


def extract_email_train_features(pickle_dir):
    """
    Extracts features from the email training dataset

    Parameters
    ----------
    pickle_dir : str
        The folder to output pickles to
    Returns
    -------
    X_train:
        A scipy sparse matrix containing the extracted features
    y_train:
        A list containing the labels for the extracted dataset
    vectorizer:
        The sklearn vectorizer for the features
    features:
        The features loaded during training
    """
    if not os.path.isdir(pickle_dir):
        os.makedirs(pickle_dir)

    print("Extracting Train Set")
    phishbench_globals.logger.info("Extracting Train Set")
    legit_path = pb_input.settings.train_legit_path()
    phish_path = pb_input.settings.train_phish_path()

    feature_list_dict_train, y_train, features = email_extraction.extract_labeled_dataset(legit_path, phish_path)

    preprocessing.clean_features(feature_list_dict_train)

    # Export features to csv
    if phishbench_globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(pickle_dir, 'features.csv')
        export_features_to_csv(feature_list_dict_train, y_train, out_path)

    # Tranform the list of dictionaries into a sparse matrix
    vectorizer = preprocessing.Vectorizer()
    x_train = vectorizer.fit_transform(feature_list_dict_train)

    joblib.dump(x_train, os.path.join(pickle_dir, "X_train_unprocessed.pkl"))

    x_train = Features_Support.Preprocessing(x_train)
    print(x_train.shape)
    return x_train, y_train, vectorizer, features


def extract_email_test_features(pickle_dir, features, vectorizer=None):
    """
    Extracts features from the email testing dataset

    Parameters
    ----------
    pickle_dir : str
        The folder to output pickles to
    features:
        The features to extract
    vectorizer :
        The vectorizer used to vectorize the training dataset

    Returns
    -------
    x_test:
        A scipy sparse matrix containing the extracted features
    y_test
        A list containing the dataset labels
    """

    if not os.path.isdir(pickle_dir):
        os.makedirs(pickle_dir)

    legit_path = pb_input.settings.test_legit_path()
    phish_path = pb_input.settings.test_phish_path()

    print("Extracting Test Set")
    phishbench_globals.logger.info('Extracting Test Set')
    feature_list_dict_test, y_test, _ = email_extraction.extract_labeled_dataset(legit_path, phish_path, features)

    preprocessing.clean_features(feature_list_dict_test)

    # Export features to csv
    if phishbench_globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(pickle_dir, 'features.csv')
        export_features_to_csv(feature_list_dict_test, y_test, out_path)

    # Tranform the list of dictionaries into a sparse matrix
    x_test = vectorizer.transform(feature_list_dict_test)

    # Use Min_Max_scaling for pre-processing the feature matrix
    x_test = Features_Support.Preprocessing(x_test)

    phishbench_globals.logger.info("Feature Extraction for testing dataset: Done!")

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
    email_train_dir = os.path.join(phishbench_globals.args.output_input_dir, "Emails_Training")
    email_test_dir = os.path.join(phishbench_globals.args.output_input_dir, "Emails_Testing")

    if phishbench_globals.config["Extraction"].getboolean("Training Dataset"):
        x_train, y_train, vectorizer, features = extract_email_train_features(email_train_dir)

        # Save features for training dataset
        joblib.dump(x_train, os.path.join(email_train_dir, "X_train.pkl"))
        joblib.dump(y_train, os.path.join(email_train_dir, "y_train.pkl"))
        joblib.dump(vectorizer, os.path.join(email_train_dir, "vectorizer.pkl"))
        if not os.path.isdir(os.path.join(email_train_dir, "features")):
            os.makedirs(os.path.join(email_train_dir, "features"))
        for feature in features:
            path = os.path.join(email_train_dir, "features", f"{feature.config_name}.pkl")
            feature.save_state(path)
        phishbench_globals.logger.info("Feature Extraction for training dataset: Done!")
    else:
        x_train = joblib.load(os.path.join(email_train_dir, "X_train.pkl"))
        y_train = joblib.load(os.path.join(email_train_dir, "y_train.pkl"))
        vectorizer = joblib.load(os.path.join(email_train_dir, "vectorizer.pkl"))
        features = email_extraction.create_new_features()
        for feature in features:
            path = os.path.join(email_train_dir, "features", f"{feature.config_name}.pkl")
            feature.load_state(path)

    if phishbench_globals.config["Extraction"]["Testing Dataset"] == "True":
        x_test, y_test = extract_email_test_features(email_test_dir, features, vectorizer)
        joblib.dump(x_test, os.path.join(email_test_dir, "X_test.pkl"))
        joblib.dump(y_test, os.path.join(email_test_dir, "y_test.pkl"))
    else:
        x_test = None
        y_test = None
    tfidf_vectorizer = None
    for feature in features:
        if isinstance(feature, email_extraction.features.EmailBodyTfidf):
            tfidf_vectorizer = feature.tfidf_vectorizer
    return x_train, y_train, x_test, y_test, vectorizer.scalar_vectorizer, tfidf_vectorizer


def get_config():
    tfidf_vec = None
    if phishbench.settings.mode() == 'Email':
        train_dir = os.path.join(phishbench_globals.args.output_input_dir, "Emails_Training")
        test_dir = os.path.join(phishbench_globals.args.output_input_dir, "Emails_Testing")
        run_tfidf = extraction_settings.feature_type_enabled(FeatureType.EMAIL_BODY) and \
                    phishbench_globals.config[FeatureType.EMAIL_BODY].getboolean("email_body_tfidf")
        if run_tfidf:
            tfidf_vec = os.path.join(train_dir, "features", "email_body_tfidf.pkl")
    else:
        train_dir = os.path.join(phishbench_globals.args.output_input_dir, "URLs_Training")
        test_dir = os.path.join(phishbench_globals.args.output_input_dir, "URLs_Testing")
        run_tfidf = extraction_settings.feature_type_enabled(FeatureType.URL_WEBSITE) and \
                    phishbench_globals.config[FeatureType.URL_WEBSITE].getboolean("website_tfidf")
        if run_tfidf:
            tfidf_vec = os.path.join(train_dir, "features", "website_tfidf.pkl")
    return train_dir, test_dir, tfidf_vec


def load_dataset():
    train_dir, test_dir, tfidf_vec = get_config()
    print(tfidf_vec)
    x_train = joblib.load(os.path.join(train_dir, "X_train.pkl"))
    y_train = joblib.load(os.path.join(train_dir, "y_train.pkl"))
    vectorizer = joblib.load(os.path.join(train_dir, "vectorizer.pkl"))
    vectorizer = vectorizer.scalar_vectorizer
    if tfidf_vec:
        tfidf_vectorizer = joblib.load(tfidf_vec)
    else:
        tfidf_vectorizer = None
    if os.path.exists(os.path.join(test_dir, "X_test.pkl")):
        x_test = joblib.load(os.path.join(test_dir, "X_test.pkl"))
        y_test = joblib.load(os.path.join(test_dir, "y_test.pkl"))
    else:
        x_test = None
        y_test = None
    return x_train, y_train, vectorizer, tfidf_vectorizer, x_test, y_test


def run_classifiers(x_train, y_train, x_test, y_test):
    folder = os.path.join(phishbench_globals.args.output_input_dir, "Classifiers")
    print("Training Classifiers")

    classifiers = classification.train_classifiers(x_train, y_train, io_dir=folder)
    classifier_performances = evaluation.evaluate_classifiers(classifiers, x_test, y_test)

    print(classifier_performances)
    classifier_performances.to_csv(os.path.join(folder, "performance.csv"), index=False)


def run_phishbench():
    if phishbench.settings.feature_extraction():
        if phishbench.settings.mode() == 'Email':
            x_train, y_train, x_test, y_test, vectorizer, tfidf_vectorizer = extract_email_features()
        else:
            x_train, y_train, x_test, y_test, vectorizer, tfidf_vectorizer = extract_url_features()
    else:
        x_train, y_train, vectorizer, tfidf_vectorizer, x_test, y_test = load_dataset()

    # Feature Selection
    if phishbench.settings.feature_selection():
        ranking_dir = os.path.join(phishbench_globals.args.output_input_dir, "Feature_Ranking")
        if not os.path.exists(ranking_dir):
            os.makedirs(ranking_dir)
        # k: Number of Best features
        num_best_features = int(phishbench_globals.config["Feature Selection"]["number of best features"])
        x_train, selection_model = Feature_Selection.Feature_Ranking(x_train, y_train, num_best_features,
                                                                     vectorizer, tfidf_vectorizer)
        # Dump model
        joblib.dump(selection_model, os.path.join(ranking_dir, "selection.pkl"))
        joblib.dump(x_train, os.path.join(ranking_dir, "X_train_processed_best_features.pkl"))

        if x_test is not None:
            x_test = selection_model.transform(x_test)
            phishbench_globals.logger.info("X_test Shape: %s", x_test.shape)
            joblib.dump(x_test, os.path.join(ranking_dir, "X_test_processed_best_features.pkl"))

    if phishbench.settings.classification():
        run_classifiers(x_train, y_train, x_test, y_test)


def main():
    # execute only if run as a script
    phishbench_globals.setup_parser()
    if phishbench_globals.args.version:
        print("PhishBench ", phishbench.__version__)
        sys.exit(0)
    phishbench_globals.initialize(phishbench_globals.args.config_file)
    answer = user_interaction.confirmation(phishbench_globals.args.ignore_confirmation)
    original = sys.stdout
    if answer:
        phishbench_globals.logger.debug("Running......")
        run_phishbench()
        phishbench_globals.logger.debug("Done!")
    sys.stdout = original
    phishbench_globals.destroy_globals()


if __name__ == "__main__":
    main()
