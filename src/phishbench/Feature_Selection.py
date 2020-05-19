import math
import os

import joblib
import numpy as np
import sklearn
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .utils import Globals


def Feature_Ranking(X, y, k):
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
        model = LogisticRegression()
        from sklearn.svm import LinearSVC
        model = LinearSVC()
        rfe = RFE(model, k, verbose=2, step=0.005)
        rfe.fit(X, y)
        X = rfe.transform(X)
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
            joblib.dump(X, "Data_Dump/Emails_Training/X_train_with_tfidf_RFE_{}.pkl".format(k))
        if urls:
            joblib.dump(X, "Data_Dump/URLs_Training/X_train_with_tfidf_RFE_{}.pkl".format(k))
        return X, rfe

    # Chi-2
    elif Globals.config["Feature Selection"]["Chi-2"] == "True":
        model = sklearn.feature_selection.SelectKBest(chi2, k)
        model.fit(X, y)
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
        X = model.transform(X)
        if emails:
            joblib.dump(X, "Data_Dump/Emails_Training/X_train_with_tfidf_Chi2_{}.pkl".format(k))
        if urls:
            joblib.dump(X, "Data_Dump/URLs_Training/X_train_with_tfidf_Chi2_{}.pkl".format(k))
        return X, model

    # Information Gain
    elif Globals.config["Feature Selection"]["Information Gain"] == "True":
        model = sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='entropy'),
                                                          threshold=-np.inf, max_features=k)
        model.fit(X, y)
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
        X = model.transform(X)
        if emails:
            joblib.dump(X, "Data_Dump/Emails_Training/X_train_with_tfidf_IG_{}.pkl".format(k))
        if urls:
            joblib.dump(X, "Data_Dump/URLs_Training/X_train_with_tfidf_IG_{}.pkl".format(k))
        return X, vectorizer

    # Gini
    elif Globals.config["Feature Selection"]["Gini"] == "True":
        model = sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='gini'), threshold=-np.inf,
                                                          max_features=k)
        model.fit(X, y)
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
        X = model.transform(X)
        if emails:
            joblib.dump(X, "Data_Dump/Emails_Training/X_train_with_tfidf_Gini_{}.pkl".format(k))
        if urls:
            joblib.dump(X, "Data_Dump/URLs_Training/X_train_with_tfidf_Gini_{}.pkl".format(k))
        return X, vectorizer


def Select_Best_Features_Testing(X, selection, k, feature_list_dict_test):
    if Globals.config["Feature Selection"]["Recursive Feature Elimination"] == "True":
        X = selection.transform(X)
        Globals.logger.info("X_Shape: {}".format(X.shape))
        return X
    elif Globals.config["Feature Selection"]["Chi-2"] == "True":
        X = selection.transform(X)
        Globals.logger.info("X_Shape: {}".format(X.shape))
        return X
    elif Globals.config["Feature Selection"]["Information Gain"] == "True":
        best_features = []
        with open("Data_Dump/Feature_Ranking/Feature_ranking_IG.txt", 'r') as f:
            for line in f.readlines():
                best_features.append(line.split(':')[0])
        new_list_dict_features = []
        for i in range(k):
            key = best_features[i]
            if "=" in key:
                key = key.split("=")[0]
            if i == 0:
                for j in range(len(feature_list_dict_test)):
                    new_list_dict_features.append({key: feature_list_dict_test[j][key]})
            else:
                for j in range(len(feature_list_dict_test)):
                    new_list_dict_features[j][key] = feature_list_dict_test[j][key]
        X = selection.transform(new_list_dict_features)
        Globals.logger.info("X_Shape: {}".format(X.shape))
        return X
    elif Globals.config["Feature Selection"]["Gini"] == "True":
        best_features = []
        with open("Data_Dump/Feature_Ranking/Feature_ranking_Gini.txt", 'r') as f:
            for line in f.readlines():
                best_features.append(line.split(':')[0])
        new_list_dict_features = []
        for i in range(k):
            key = best_features[i]
            # Globals.logger.info("key: {}".format(key))
            if "=" in key:
                key = key.split("=")[0]
            if i == 0:
                for j in range(len(feature_list_dict_test)):
                    new_list_dict_features.append({key: feature_list_dict_test[j][key]})
            else:
                for j in range(len(feature_list_dict_test)):
                    new_list_dict_features[j][key] = feature_list_dict_test[j][key]
        Globals.logger.info(new_list_dict_features)
        Globals.logger.info("new_list_dict_features shape: {}".format(len(new_list_dict_features[0])))
        X = selection.transform(new_list_dict_features)
        return X
