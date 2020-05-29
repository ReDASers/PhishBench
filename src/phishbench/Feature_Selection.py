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


def Feature_Ranking(features, target, num_features, vectorizer, vectorizer_tfidf=None):
    print('Feature Ranking Started')

    feature_ranking_folder = os.path.join(Globals.args.output_input_dir, 'Feature_Ranking')
    if not os.path.exists(feature_ranking_folder):
        os.makedirs(feature_ranking_folder)

    if vectorizer_tfidf:
        features_list = (vectorizer.get_feature_names()) + (vectorizer_tfidf.get_feature_names())
    else:
        features_list = (vectorizer.get_feature_names())

    # RFE
    if Globals.config["Feature Selection"]["Recursive Feature Elimination"] == "True":
        selection_model = RFE(LinearSVC(), num_features, verbose=2, step=0.005)
        selection_model.fit(features, target)
        res = dict(zip(features_list, selection_model.ranking_))
        report_name = "Feature_ranking_rfe.txt"

    # Chi-2
    elif Globals.config["Feature Selection"]["Chi-2"] == "True":
        selection_model = sklearn.feature_selection.SelectKBest(chi2, num_features)
        selection_model.fit(features, target)
        res = dict(zip(features_list, selection_model.scores_))
        report_name = "Feature_ranking_chi2.txt"

    # Information Gain
    elif Globals.config["Feature Selection"]["Information Gain"] == "True":
        selection_model = sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='entropy'),
                                                          threshold=-np.inf, max_features=num_features)
        selection_model.fit(features, target)
        # dump Feature Selection in a file
        res = dict(zip(features_list, selection_model.estimator_.feature_importances_))
        report_name = "Feature_ranking_IG.txt"

    # Gini
    elif Globals.config["Feature Selection"]["Gini"] == "True":
        selection_model = sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='gini'), threshold=-np.inf,
                                                          max_features=num_features)
        selection_model.fit(features, target)
        res = dict(zip(features_list, selection_model.estimator_.feature_importances_))
        report_name = "Feature_ranking_Gini.txt"

    for key, value in res.items():
        if math.isnan(res[key]):
            res[key] = 0
    sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join(feature_ranking_folder, report_name), 'w') as f:
        for (key, value) in sorted_d:
            f.write("{}: {}\n".format(key, value))

    # create new feature set with the best k features
    features = selection_model.transform(features)

    return features, selection_model
