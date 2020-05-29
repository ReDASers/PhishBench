import math
import os

import joblib
import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier

from .utils import Globals


def Feature_Ranking(features, target, num_features):
    print('Feature Ranking Started')

    feature_ranking_folder = os.path.join(Globals.args.output_input_dir, 'Feature_Ranking')
    if not os.path.exists(feature_ranking_folder):
        os.makedirs(feature_ranking_folder)
    email_train_dir = os.path.join(Globals.args.output_input_dir, 'Emails_Training')
    url_train_dir = os.path.join(Globals.args.output_input_dir, 'URLs_Training')

    if Globals.config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
        emails = True
        urls = False
        vectorizer = joblib.load(os.path.join(email_train_dir, 'vectorizer.pkl'))
        if Globals.config["Feature Selection"]["with Tfidf"] == "True":
            vectorizer_tfidf = joblib.load(os.path.join(email_train_dir, "tfidf_vectorizer.pkl"))

    elif Globals.config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
        urls = True
        emails = False
        vectorizer = joblib.load(os.path.join(url_train_dir, "vectorizer.pkl"))
        if Globals.config["Feature Selection"]["with Tfidf"] == "True":
            vectorizer_tfidf = joblib.load(os.path.join(url_train_dir, "tfidf_vectorizer.pkl"))

    # RFE
    if Globals.config["Feature Selection"]["Recursive Feature Elimination"] == "True":
        selection_model = RFE(LinearSVC(), num_features, verbose=2, step=0.005)
        selection_model.fit(features, target)

        if Globals.config["Feature Selection"]["with Tfidf"] == "True":
            features_list = (vectorizer.get_feature_names()) + (vectorizer_tfidf.get_feature_names())
        else:
            features_list = (vectorizer.get_feature_names())
        res = dict(zip(features_list, selection_model.ranking_))
        sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)

        with open(os.path.join(feature_ranking_folder, "Feature_ranking_rfe.txt"), 'w') as f:
            for (key, value) in sorted_d:
                f.write("{}: {}\n".format(key, value))

        outfile_name = "X_train_with_tfidf_RFE_{}.pkl".format(num_features)

    # Chi-2
    elif Globals.config["Feature Selection"]["Chi-2"] == "True":
        selection_model = sklearn.feature_selection.SelectKBest(chi2, num_features)
        selection_model.fit(features, target)
        if Globals.config["Feature Selection"]["with Tfidf"] == "True":
            features_list = (vectorizer.get_feature_names()) + (vectorizer_tfidf.get_feature_names())
        else:
            features_list = (vectorizer.get_feature_names())
        res = dict(zip(features_list, selection_model.scores_))
        for key, value in res.items():
            if math.isnan(res[key]):
                res[key] = 0
        sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)

        with open(os.path.join(feature_ranking_folder, "Feature_ranking_chi2.txt"), 'w') as f:
            for (key, value) in sorted_d:
                f.write("{}: {}\n".format(key, value))
        features = selection_model.transform(features)
        outfile_name = "X_train_with_tfidf_Chi2_{}.pkl".format(num_features)

    # Information Gain
    elif Globals.config["Feature Selection"]["Information Gain"] == "True":
        selection_model = sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='entropy'),
                                                          threshold=-np.inf, max_features=num_features)
        selection_model.fit(features, target)
        # dump Feature Selection in a file
        if Globals.config["Feature Selection"]["with Tfidf"] == "True":
            features_list = (vectorizer.get_feature_names()) + (vectorizer_tfidf.get_feature_names())
        else:
            features_list = (vectorizer.get_feature_names())
        res = dict(zip(features_list, selection_model.estimator_.feature_importances_))
        for key, value in res.items():
            if math.isnan(res[key]):
                res[key] = 0
        sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
        with open(os.path.join(feature_ranking_folder, "Feature_ranking_IG.txt"), 'w') as f:
            for (key, value) in sorted_d:
                f.write("{}: {}\n".format(key, value))

        outfile_name = "X_train_with_tfidf_IG_{}.pkl".format(num_features)

    # Gini
    elif Globals.config["Feature Selection"]["Gini"] == "True":
        selection_model = sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='gini'), threshold=-np.inf,
                                                          max_features=num_features)
        selection_model.fit(features, target)
        if Globals.config["Feature Selection"]["with Tfidf"] == "True":
            features_list = (vectorizer.get_feature_names()) + (vectorizer_tfidf.get_feature_names())
        else:
            features_list = (vectorizer.get_feature_names())
        res = dict(zip(features_list, selection_model.estimator_.feature_importances_))
        for key, value in res.items():
            if math.isnan(res[key]):
                res[key] = 0
        sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
        with open(os.path.join(feature_ranking_folder, "Feature_ranking_Gini.txt"), 'w') as f:
            for (key, value) in sorted_d:
                f.write("{}: {}\n".format(key, value))
        outfile_name = "X_train_with_tfidf_Gini_{}.pkl".format(num_features)

    # create new feature set with the best k features
    features = selection_model.transform(features)

    if emails:
        joblib.dump(features, os.path.join(email_train_dir, outfile_name))
    if urls:
        joblib.dump(features, os.path.join(url_train_dir, outfile_name))

    return features, selection_model
