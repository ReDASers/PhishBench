"""
Implementations for feature sampling methods
"""
import imblearn.under_sampling as under_sampling
import imblearn.over_sampling as over_sampling
import imblearn.combine as combine


def condensed_nearest_neighbor(features, labels):
    """
    Undersamples using Condensed Nearest Neighbor method

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = under_sampling.CondensedNearestNeighbour()
    return sampler.fit_sample(features, labels)


def edited_nearest_neighbor(features, labels):
    """
    Undersamples using Edited Nearest Neighbor method

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = under_sampling.EditedNearestNeighbours()
    return sampler.fit_sample(features, labels)


def repeated_edited_nearest_neighbor(features, labels):
    """
    Undersamples using Repeated Edited Nearest Neighbor method

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = under_sampling.RepeatedEditedNearestNeighbours()
    return sampler.fit_sample(features, labels)


def all_knn(features, labels):
    """
    Undersamples using AllKNN method

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = under_sampling.AllKNN()
    return sampler.fit_sample(features, labels)


def instance_hardness_threshold(features, labels):
    """
    Undersamples using instance hardness threshold method

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = under_sampling.InstanceHardnessThreshold()
    return sampler.fit_sample(features, labels)


def near_miss(features, labels):
    """
    Undersamples using NearMiss method

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = under_sampling.NearMiss()
    return sampler.fit_sample(features, labels)


def neighborhood_cleaning_rule(features, labels):
    """
    Undersamples using neighborhood cleaning rule

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = under_sampling.NeighbourhoodCleaningRule()
    return sampler.fit_sample(features, labels)


def one_sided_selection(features, labels):
    """
    Undersamples using one sided selection method

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = under_sampling.OneSidedSelection()
    return sampler.fit_sample(features, labels)


def random_undersampling(features, labels):
    """
    Undersamples by randomly picking from the majority class

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = under_sampling.RandomUnderSampler()
    return sampler.fit_sample(features, labels)


def tomek_links(features, labels):
    """
    Undersamples using Tomek's Link method

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = under_sampling.TomekLinks()
    return sampler.fit_sample(features, labels)


def adasyn(features, labels):
    """
    Oversamples using ADASYN

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = over_sampling.ADASYN()
    try:
        return sampler.fit_sample(features, labels)
    except RuntimeError as e:
        # Not any neighbors belong to the majority class
        print(e)
        return None


def random_oversampling(features, labels):
    """
    Oversamples by randomly selecting from the minority class with replacement

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = over_sampling.RandomOverSampler()
    return sampler.fit_sample(features, labels)


def smote(features, labels):
    """
    Oversamples using SMOTE

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = over_sampling.SMOTE()
    return sampler.fit_sample(features, labels)


def borderline_smote(features, labels):
    """
    Oversamples using Borderline SMOTE

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = over_sampling.BorderlineSMOTE()
    return sampler.fit_sample(features, labels)


def smote_enn(features, labels):
    """
    Oversamples using SMOTE and cleans using ENN

    Parameters
    ==========
    features:
        A matrix containing the features of the dataset
    labels:
        The labels for each sample in features

    Returns
    =======
    sampled_features:
        The features of the sampled dataset
    sampled_labels:
        The labels of the sampled dataset
    """
    sampler = combine.SMOTEENN()
    return sampler.fit_sample(features, labels)


METHODS = {
    'condensed nearest neighbor': condensed_nearest_neighbor,
    'edited nearest neighbor': edited_nearest_neighbor,
    'repeated edited nearest neighbor': repeated_edited_nearest_neighbor,
    'all knn': all_knn,
    'instance hardness threshold': instance_hardness_threshold,
    'near miss': near_miss,
    'neighborhood cleaning rule': neighborhood_cleaning_rule,
    'random undersampling': random_undersampling,
    'tomek links': tomek_links,
    'adasyn': adasyn,
    'random oversampling': random_oversampling,
    'SMOTE': smote,
    'borderline SMOTE': borderline_smote,
    'smote and enn': smote_enn
}
