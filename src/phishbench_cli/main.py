import os
import sys
from types import ModuleType
from typing import List, Dict

import joblib
import pandas as pd

import phishbench
import phishbench.Features_Support as Features_Support
import phishbench.classification as classification
import phishbench.evaluation as evaluation
import phishbench.feature_extraction.email as email_extraction
import phishbench.feature_extraction.url as url_extraction
import phishbench.feature_preprocessing as preprocessing
import phishbench.feature_preprocessing.feature_selection.settings
import phishbench.input as pb_input
import phishbench.settings
from phishbench.feature_extraction import settings as extraction_settings
from phishbench.feature_extraction.reflection import FeatureClass, FeatureType
from phishbench.utils import phishbench_globals
from phishbench_cli import user_interaction


def export_features_to_csv(features: List[Dict], y: List[int], file_path: str):
    """
    Exports raw features to csv

    Parameters
    ----------
    features : List[Dict]
        A list of dicts with each dict containing the features of a single data point
    y: List[int]
        The labels for each datapoint. This should have the same length as features
    file_path: str
        The file to save the csv to
    """
    df = pd.DataFrame(features)
    df['is_phish'] = y
    df.to_csv(file_path, index=False)


def extract_train_features(pickle_dir: str,
                           features: List[FeatureClass],
                           extraction_module: ModuleType):
    """
    Extracts features from the training dataset

    Parameters
    ----------
    pickle_dir : str
        The folder to output pickles to
    features:
        The features to extract
    extraction_module: ModuleType
        Either `email_extraction` or `url_extraction`

    Returns
    -------
    x_train:
        A scipy sparse matrix containing the extracted features
    y_train:
        A list containing the labels for the extracted dataset
    vectorizer:
        The sklearn vectorizer for the features
    """
    if not os.path.isdir(pickle_dir):
        os.makedirs(pickle_dir)
    if not hasattr(extraction_module, 'extract_features_list'):
        raise ValueError('extraction_module must be an extraction module')

    print("Loading Train Set")
    phishbench_globals.logger.info("Extracting Train Set")

    emails, y_train = pb_input.read_train_set(extraction_settings.download_url_flag())

    print("Extracting Features")
    for feature in features:
        feature.fit(emails, y_train)
    feature_list_dict_train = extraction_module.extract_features_list(emails, features)

    print("Cleaning features")
    preprocessing.clean_features(feature_list_dict_train)

    # Export features to csv
    if phishbench_globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(pickle_dir, 'train_features.csv')
        export_features_to_csv(feature_list_dict_train, y_train, out_path)

    # Tranform the list of dictionaries into a sparse matrix
    vectorizer = preprocessing.Vectorizer()
    x_train = vectorizer.fit_transform(feature_list_dict_train)

    joblib.dump(x_train, os.path.join(pickle_dir, "X_train_unprocessed.pkl"))

    x_train = Features_Support.Preprocessing(x_train)
    print(x_train.shape)
    return x_train, y_train, vectorizer


def extract_test_features(pickle_dir: str,
                          vectorizer: preprocessing.Vectorizer,
                          features: List[FeatureClass],
                          extraction_module: ModuleType):
    """
    Extracts features from the testing dataset

    Parameters
    ----------
    pickle_dir: str
        The folder to output pickles to
    features:
        The features to extract
    vectorizer: List[FeatureClass]
        The vectorizer used to vectorize the training dataset
    extraction_module: ModuleType
        Either `email_extraction` or `url_extraction`

    Returns
    -------
    x_test:
        A scipy sparse matrix containing the extracted features
    y_test:
        A list containing the dataset labels
    """

    if not os.path.isdir(pickle_dir):
        os.makedirs(pickle_dir)
    if not hasattr(extraction_module, 'extract_features_list'):
        raise ValueError('extraction_module must be an extraction module')

    print("Loading Test Set")
    emails, y_test = pb_input.read_test_set(extraction_settings.download_url_flag())

    print("Extracting Features")
    feature_list_dict_test = extraction_module.extract_features_list(emails, features)

    print("Cleaning features")
    preprocessing.clean_features(feature_list_dict_test)

    # Export features to csv
    if phishbench_globals.config['Features Export'].getboolean('csv'):
        out_path = os.path.join(pickle_dir, 'test_features.csv')
        export_features_to_csv(feature_list_dict_test, y_test, out_path)

    # Transform the list of dictionaries into a sparse matrix
    x_test = vectorizer.transform(feature_list_dict_test)

    # Use Min_Max_scaling for pre-processing the feature matrix
    x_test = Features_Support.Preprocessing(x_test)

    return x_test, y_test


def extract_features(extraction_module: ModuleType):
    """
    Extracts features. If PhishBench is configured to only extract features from a test dataset, this function will
    load pre-extracted training data from disk.

    Parameters
    ----------
    extraction_module: ModuleType
        Either `email_extraction` or `url_extraction`

    Returns
    -------
    x_train:
        A scipy sparse matrix containing the extracted features
    y_train:
        A list containing the labels for the extracted dataset
    x_test:
        A scipy sparse matrix containing the extracted features
    y_test:
        A list containing the dataset labels
    vectorizer:
        The sklearn vectorizer for the features
    tfidf_vectorizer:
        The TF-IDF vectorizer used to generate TFIDF vectors. None if TF-IDF is not run
    """
    pickle_dir = os.path.join(phishbench_globals.output_dir, "Features")

    if not hasattr(extraction_module, 'extract_features_list'):
        raise ValueError('extraction_module must be an extraction module')
    if not hasattr(extraction_module, 'create_new_features'):
        raise ValueError('extraction_module must be an extraction module')

    features = extraction_module.create_new_features()

    if phishbench_globals.config["Extraction"].getboolean("Training Dataset"):
        x_train, y_train, vectorizer = extract_train_features(pickle_dir, features, extraction_module)

        # dump features and labels and vectorizers
        joblib.dump(x_train, os.path.join(pickle_dir, "X_train.pkl"))
        joblib.dump(y_train, os.path.join(pickle_dir, "y_train.pkl"))
        joblib.dump(vectorizer, os.path.join(pickle_dir, "vectorizer.pkl"))
        if not os.path.isdir(os.path.join(pickle_dir, "features")):
            os.makedirs(os.path.join(pickle_dir, "features"))
        for feature in features:
            path = os.path.join(pickle_dir, "features", f"{feature.config_name}.pkl")
            feature.save_state(path)
        phishbench_globals.logger.info("Feature Extraction for training dataset: Done!")
    else:
        # if training was done in another instance of the platform then load the necessary files
        x_train = joblib.load(os.path.join(pickle_dir, "X_train.pkl"))
        y_train = joblib.load(os.path.join(pickle_dir, "y_train.pkl"))
        vectorizer = joblib.load(os.path.join(pickle_dir, "vectorizer.pkl"))
        for feature in features:
            path = os.path.join(pickle_dir, "features", f"{feature.config_name}.pkl")
            feature.load_state(path)

    if phishbench_globals.config["Extraction"].getboolean("Testing Dataset"):
        x_test, y_test = extract_test_features(pickle_dir, vectorizer, features, extraction_module)

        joblib.dump(x_test, os.path.join(pickle_dir, "X_test.pkl"))
        joblib.dump(y_test, os.path.join(pickle_dir, "y_test.pkl"))
        phishbench_globals.logger.info("Feature Extraction for testing dataset: Done!")
    else:
        x_test = None
        y_test = None

    tfidf_vectorizer = None
    for feature in features:
        if hasattr(feature, 'tfidf_vectorizer'):
            tfidf_vectorizer = feature.tfidf_vectorizer

    return x_train, y_train, x_test, y_test, vectorizer.scalar_vectorizer, tfidf_vectorizer


def get_tfidf_path():
    """
    Gets the path to the tfidf_vectorizer
    """
    train_dir = os.path.join(phishbench_globals.output_dir, "Features")
    if phishbench.settings.mode() == 'Email':
        run_tfidf = extraction_settings.feature_type_enabled(FeatureType.EMAIL_BODY) and \
                    phishbench_globals.config[FeatureType.EMAIL_BODY.value].getboolean("email_body_tfidf")
        if run_tfidf:
            return os.path.join(train_dir, "features", "email_body_tfidf.pkl")
    else:
        run_tfidf = extraction_settings.feature_type_enabled(FeatureType.URL_WEBSITE) and \
                    phishbench_globals.config[FeatureType.URL_WEBSITE.value].getboolean("website_tfidf")
        if run_tfidf:
            return os.path.join(train_dir, "features", "website_tfidf.pkl")
    return None


def load_features_from_disk():
    """
    Loads pre-extracted features from disk

    Returns
    -------
    X_train:
        A scipy sparse matrix containing the extracted features
    y_train:
        A list containing the labels for the extracted dataset
    x_test:
        A scipy sparse matrix containing the extracted features
    y_test:
        A list containing the dataset labels
    vectorizer:
        The sklearn vectorizer for the features
    tfidf_vectorizer:
        The TF-IDF vectorizer used to generate TFIDF vectors. None if TF-IDF is not run
    """
    train_dir = os.path.join(phishbench_globals.output_dir, "Features")
    tfidf_vec = get_tfidf_path()

    x_train = joblib.load(os.path.join(train_dir, "X_train.pkl"))
    y_train = joblib.load(os.path.join(train_dir, "y_train.pkl"))
    vectorizer = joblib.load(os.path.join(train_dir, "vectorizer.pkl"))
    vectorizer = vectorizer.scalar_vectorizer
    if tfidf_vec:
        tfidf_vectorizer = joblib.load(tfidf_vec)
    else:
        tfidf_vectorizer = None
    if os.path.exists(os.path.join(train_dir, "X_test.pkl")):
        x_test = joblib.load(os.path.join(train_dir, "X_test.pkl"))
        y_test = joblib.load(os.path.join(train_dir, "y_test.pkl"))
    else:
        x_test = None
        y_test = None
    return x_train, y_train, vectorizer, tfidf_vectorizer, x_test, y_test


def run_classifiers(x_train, y_train, x_test, y_test, folder):
    """
    Runs and evaluates the classifiers

    Parameters
    ----------
    x_train:
        A scipy sparse matrix containing the extracted features
    y_train:
        A list containing the labels for the extracted dataset
    x_test:
        A scipy sparse matrix containing the extracted features
    y_test:
        A list containing the dataset labels

    Returns
    -------
    """
    classifiers = classification.train_classifiers(x_train, y_train, io_dir=folder)
    classifier_performances = evaluation.evaluate_classifiers(classifiers, x_test, y_test)

    return classifier_performances


def run_phishbench():
    """
    Runs the PhishBench basic experiment
    """
    if phishbench.settings.feature_extraction():
        if phishbench.settings.mode() == 'Email':
            x_train, y_train, x_test, y_test, vectorizer, tfidf_vectorizer = extract_features(email_extraction)
        else:
            x_train, y_train, x_test, y_test, vectorizer, tfidf_vectorizer = extract_features(url_extraction)
    else:
        x_train, y_train, vectorizer, tfidf_vectorizer, x_test, y_test = load_features_from_disk()

    if tfidf_vectorizer:
        feature_names = (vectorizer.get_feature_names()) + (tfidf_vectorizer.get_feature_names())
    else:
        feature_names = (vectorizer.get_feature_names())

    train_samples = {
        'None': (x_train, y_train)
    }
    if phishbench.settings.dataset_balancing():
        train_samples.update(preprocessing.balancing.run_sampling(x_train, y_train))

    x_train_dict2 = {}
    x_test_dict2 = {}
    y_train_dict = {}
    for balancing_method in train_samples:
        x_train, y_train = train_samples[balancing_method]
        y_train_dict[balancing_method] = y_train
        # Feature Selection
        if phishbench.settings.feature_selection():
            output_dir = os.path.join(phishbench_globals.output_dir, "Feature Selection", balancing_method)
            # x_test should be the same no matter the sampling method
            x_train_dict, x_test_dict = preprocessing.feature_selection.\
                run_feature_extraction(x_train, x_test, y_train, feature_names, output_dir)
        else:
            x_train_dict = {
                'None': x_train
            }
            x_test_dict = {
                'None': x_test
            }
        x_train_dict2[balancing_method] = x_train_dict
        x_test_dict2[balancing_method] = x_test_dict

    for key in x_train_dict2:
        print(key)
        print(x_train_dict2[key].keys())

    if phishbench.settings.classification():
        classification_dir = os.path.join(phishbench_globals.output_dir, "Classifiers")
        classifier_performances = pd.DataFrame()
        for balancing_method in x_train_dict2.keys():
            x_train_dict = x_train_dict2[balancing_method]
            x_test_dict = x_test_dict2[balancing_method]
            y_train = y_train_dict[balancing_method]
            for selection_method in x_train_dict.keys():
                x_train = x_train_dict[selection_method]
                x_test = x_test_dict[selection_method]
                folder = os.path.join(classification_dir, balancing_method, selection_method)
                method_performances = run_classifiers(x_train, y_train, x_test, y_test, folder)
                method_performances['Balancing Method'] = balancing_method
                method_performances['Selection Method'] = selection_method
                classifier_performances = classifier_performances.append(method_performances)

        columns: List = classifier_performances.columns.tolist()
        columns.remove('Balancing Method')
        columns.remove('Selection Method')
        columns.insert(0, 'Selection Method')
        columns.insert(0, 'Balancing Method')
        classifier_performances = classifier_performances.reindex(columns=columns)
        print(classifier_performances)
        out_csv = os.path.join(classification_dir, "performance.csv")
        classifier_performances.to_csv(out_csv, index=False)


def main():
    # execute only if run as a script
    phishbench_globals.parse_args()
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
