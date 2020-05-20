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
    # RFE
    if not os.path.exists("Data_Dump/Feature_Ranking"):
        os.makedirs("Data_Dump/Feature_Ranking")
    if Globals.config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
        emails = True
        urls = False
        vectorizer = joblib.load("Data_Dump/Emails_Training/vectorizer.pkl")
        if Globals.config["Feature Selection"]["with Tfidf"] == "True":
            vectorizer_tfidf = joblib.load("Data_Dump/Emails_Training/tfidf_vectorizer.pkl")
    elif Globals.config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
        urls = True
        emails = False
        vectorizer = joblib.load("Data_Dump/URLs_Training/vectorizer.pkl")
        if Globals.config["Feature Selection"]["with Tfidf"] == "True":
            vectorizer_tfidf = joblib.load("Data_Dump/URLs_Training/tfidf_vectorizer.pkl")
    if Globals.config["Feature Selection"]["Recursive Feature Elimination"] == "True":
        rfe = RFE(LinearSVC(), num_features, verbose=2, step=0.005)
        rfe.fit(features, target)
        features = rfe.transform(features)
        if Globals.config["Feature Selection"]["with Tfidf"] == "True":
            features_list = (vectorizer.get_feature_names()) + (vectorizer_tfidf.get_feature_names())
        else:
            features_list = (vectorizer.get_feature_names())
        res = dict(zip(features_list, rfe.ranking_))
        sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
        with open("Data_Dump/Feature_Ranking/Feature_ranking_rfe.txt", 'w') as f:
            for (key, value) in sorted_d:
                f.write("{}: {}\n".format(key, value))
        if emails:
            joblib.dump(features, "Data_Dump/Emails_Training/X_train_with_tfidf_RFE_{}.pkl".format(num_features))
        if urls:
            joblib.dump(features, "Data_Dump/URLs_Training/X_train_with_tfidf_RFE_{}.pkl".format(num_features))
        return features, rfe

    # Chi-2
    elif Globals.config["Feature Selection"]["Chi-2"] == "True":
        model = sklearn.feature_selection.SelectKBest(chi2, num_features)
        model.fit(features, target)
        if Globals.config["Feature Selection"]["with Tfidf"] == "True":
            features_list = (vectorizer.get_feature_names()) + (vectorizer_tfidf.get_feature_names())
        else:
            features_list = (vectorizer.get_feature_names())
        res = dict(zip(features_list, model.scores_))
        for key, value in res.items():
            if math.isnan(res[key]):
                res[key] = 0
        sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
        with open("Data_Dump/Feature_Ranking/Feature_ranking_chi2.txt", 'w') as f:
            for (key, value) in sorted_d:
                f.write("{}: {}\n".format(key, value))
        features = model.transform(features)
        if emails:
            joblib.dump(features, "Data_Dump/Emails_Training/X_train_with_tfidf_Chi2_{}.pkl".format(num_features))
        if urls:
            joblib.dump(features, "Data_Dump/URLs_Training/X_train_with_tfidf_Chi2_{}.pkl".format(num_features))
        return features, model

    # Information Gain
    elif Globals.config["Feature Selection"]["Information Gain"] == "True":
        model = sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='entropy'),
                                                          threshold=-np.inf, max_features=num_features)
        model.fit(features, target)
        # dump Feature Selection in a file
        if Globals.config["Feature Selection"]["with Tfidf"] == "True":
            features_list = (vectorizer.get_feature_names()) + (vectorizer_tfidf.get_feature_names())
        else:
            features_list = (vectorizer.get_feature_names())
        res = dict(zip(features_list, model.estimator_.feature_importances_))
        for key, value in res.items():
            if math.isnan(res[key]):
                res[key] = 0
        sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
        with open("Data_Dump/Feature_Ranking/Feature_ranking_IG.txt", 'w') as f:
            for (key, value) in sorted_d:
                f.write("{}: {}\n".format(key, value))
        # create new model with the best k features
        features = model.transform(features)
        if emails:
            joblib.dump(features, "Data_Dump/Emails_Training/X_train_with_tfidf_IG_{}.pkl".format(num_features))
        if urls:
            joblib.dump(features, "Data_Dump/URLs_Training/X_train_with_tfidf_IG_{}.pkl".format(num_features))
        return features, model

    # Gini
    elif Globals.config["Feature Selection"]["Gini"] == "True":
        model = sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='gini'), threshold=-np.inf,
                                                          max_features=num_features)
        model.fit(features, target)
        if Globals.config["Feature Selection"]["with Tfidf"] == "True":
            features_list = (vectorizer.get_feature_names()) + (vectorizer_tfidf.get_feature_names())
        else:
            features_list = (vectorizer.get_feature_names())
        res = dict(zip(features_list, model.estimator_.feature_importances_))
        sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
        for key, value in res.items():
            if math.isnan(res[key]):
                res[key] = 0
        sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
        with open("Data_Dump/Feature_Ranking/Feature_ranking_Gini.txt", 'w') as f:
            for (key, value) in sorted_d:
                f.write("{}: {}\n".format(key, value))
        # create new model with the best k features
        features = model.transform(features)
        if emails:
            joblib.dump(features, "Data_Dump/Emails_Training/X_train_with_tfidf_Gini_{}.pkl".format(num_features))
        if urls:
            joblib.dump(features, "Data_Dump/URLs_Training/X_train_with_tfidf_Gini_{}.pkl".format(num_features))
        return features, model
