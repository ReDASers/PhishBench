import math
import os

from .feature_preprocessing.feature_selection import chi_squared, gini, information_gain, rfe
from .utils import phishbench_globals


def Feature_Ranking(features, target, num_features, vectorizer, vectorizer_tfidf=None):
    print('Feature Ranking Started')

    num_features = min(num_features, features.shape[1])
    feature_ranking_folder = os.path.join(phishbench_globals.args.output_input_dir, 'Feature_Ranking')
    if not os.path.exists(feature_ranking_folder):
        os.makedirs(feature_ranking_folder)

    if vectorizer_tfidf:
        features_list = (vectorizer.get_feature_names()) + (vectorizer_tfidf.get_feature_names())
    else:
        features_list = (vectorizer.get_feature_names())

    # RFE
    if phishbench_globals.config["Feature Selection"]["Recursive Feature Elimination"] == "True":
        selection_model, ranking = rfe(features, target, num_features)
        report_name = "Feature_ranking_rfe.txt"

    # Chi-2
    elif phishbench_globals.config["Feature Selection"]["Chi-2"] == "True":
        selection_model, ranking = chi_squared(features, target, num_features)
        report_name = "Feature_ranking_chi2.txt"

    # Information Gain
    elif phishbench_globals.config["Feature Selection"]["Information Gain"] == "True":
        selection_model, ranking = information_gain(features, target, num_features)
        report_name = "Feature_ranking_IG.txt"

    # Gini
    elif phishbench_globals.config["Feature Selection"]["Gini"] == "True":
        selection_model, ranking = gini(features, target, num_features)
        report_name = "Feature_ranking_Gini.txt"
    else:
        raise RuntimeError("At least one feature selection method must be enabled.")

    ranking = [0 if math.isnan(x) else x for x in ranking]
    res = sorted(zip(features_list, ranking), key=lambda x: x[1], reverse=True)

    report_name = os.path.join(feature_ranking_folder, report_name)
    with open(report_name, 'w', errors="ignore") as f:
        for feature_name, rank in res:
            f.write(f"{feature_name}: {rank}\n")

    # create new feature set with the best k features
    features = selection_model.transform(features)

    return features, selection_model
