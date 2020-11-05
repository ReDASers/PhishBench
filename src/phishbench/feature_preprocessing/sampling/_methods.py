"""
Implementations for feature sampling methods
"""
import imblearn.under_sampling as under_sampling
import imblearn.over_sampling as over_sampling
import imblearn.combine as combine


def condensed_nearest_neighbor(features, labels):
    sampler = under_sampling.CondensedNearestNeighbour()
    return sampler.fit_sample(features, labels)


def edited_nearest_neighbor(features, labels):
    sampler = under_sampling.EditedNearestNeighbours()
    return sampler.fit_sample(features, labels)


def repeated_edited_nearest_neighbor(features, labels):
    sampler = under_sampling.RepeatedEditedNearestNeighbours()
    return sampler.fit_sample(features, labels)


def all_knn(features, labels):
    sampler = under_sampling.AllKNN()
    return sampler.fit_sample(features, labels)


def instance_hardness_threshold(features, labels):
    sampler = under_sampling.InstanceHardnessThreshold()
    return sampler.fit_sample(features, labels)


def near_miss(features, labels):
    sampler = under_sampling.NearMiss()
    return sampler.fit_sample(features, labels)


def neighborhood_cleaning_rule(features, labels):
    sampler = under_sampling.NeighbourhoodCleaningRule()
    return sampler.fit_sample(features, labels)


def one_sided_selection(features, labels):
    sampler = under_sampling.OneSidedSelection()
    return sampler.fit_sample(features, labels)


def random_undersampling(features, labels):
    sampler = under_sampling.RandomUnderSampler()
    return sampler.fit_sample(features, labels)


def tomek_links(features, labels):
    sampler = under_sampling.TomekLinks()
    return sampler.fit_sample(features, labels)


def adasyn(features, labels):
    sampler = over_sampling.ADASYN()
    return sampler.fit_sample(features, labels)


def random_oversampling(features, labels):
    sampler = over_sampling.RandomOverSampler()
    return sampler.fit_sample(features, labels)


def smote(features, labels):
    sampler = over_sampling.SMOTE()
    return sampler.fit_sample(features, labels)


def borderline_smote(features, labels):
    sampler = over_sampling.BorderlineSMOTE()
    return sampler.fit_sample(features, labels)


def smote_enn(features, labels):
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
