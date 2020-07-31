import imblearn
import imblearn.over_sampling
import imblearn.under_sampling
from sklearn.datasets import load_svmlight_file

from ..utils import globals


############### imbalanced learning
#### Condensed Nearest Neighbour
def CondensedNearestNeighbour(X, y):
    # X, y = load_svmlight_file(file)
    cnn = imblearn.under_sampling.CondensedNearestNeighbour(random_state=None, n_neighbors=3, n_jobs=-1)
    x_res, y_res = cnn.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


#### Edited Nearest Neighbours
def EditedNearestNeighbours(X, y):
    # X, y = load_svmlight_file(file)
    enn = imblearn.under_sampling.EditedNearestNeighbours(random_state=None, n_neighbors=3, n_jobs=-1)
    x_res, y_res = enn.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


#### Repeated Edited Nearest Neighbour
def RepeatedEditedNearestNeighbour(X, y):
    # X, y = load_svmlight_file(file)
    renn = imblearn.under_sampling.RepeatedEditedNearestNeighbours(n_jobs=-1)
    x_res, y_res = renn.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def AllKNN(X, y):
    # X, y = load_svmlight_file(file)
    allknn = imblearn.under_sampling.AllKNN(sampling_strategy='auto', return_indices=False, random_state=None,
                                            n_neighbors=3, kind_sel='all', allow_minority=False, n_jobs=8, ratio=None)
    x_res, y_res = allknn.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def InstanceHardnessThreshold(X, y):
    # X, y = load_svmlight_file(file)
    InstanceHardnessThreshold = imblearn.under_sampling.InstanceHardnessThreshold(estimator=None, ratio='auto',
                                                                                  return_indices=False,
                                                                                  random_state=None, cv=5, n_jobs=-1)
    x_res, y_res = InstanceHardnessThreshold.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def NearMiss(X, y):
    # X, y = load_svmlight_file(file)
    nearmiss = imblearn.under_sampling.NearMiss(ratio='auto', return_indices=False,
                                                random_state=None, version=1, size_ngh=None, n_neighbors=3,
                                                ver3_samp_ngh=None, n_neighbors_ver3=3, n_jobs=-1)
    x_res, y_res = nearmiss.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def NeighbourhoodCleaningRule(X, y):
    # X, y = load_svmlight_file(file)
    cleaningrule = imblearn.under_sampling.NeighbourhoodCleaningRule(ratio='auto', return_indices=False,
                                                                     random_state=None,
                                                                     size_ngh=None, n_neighbors=3, kind_sel='all',
                                                                     threshold_cleaning=0.5, n_jobs=-1)
    x_res, y_res = cleaningrule.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def OneSidedSelection(X, y):
    # X, y = load_svmlight_file(file)
    oneside = imblearn.under_sampling.OneSidedSelection(n_jobs=-1)
    x_res, y_res = oneside.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def RandomUnderSampler(X, y):
    # X, y = load_svmlight_file(file)
    random = imblearn.under_sampling.RandomUnderSampler(ratio='auto', return_indices=False, random_state=None,
                                                        replacement=False)
    x_res, y_res = random.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def TomekLinks(X, y):
    # X, y = load_svmlight_file(file)
    tomeklinks = imblearn.under_sampling.TomekLinks(ratio='auto', return_indices=False, random_state=None, n_jobs=-1)
    x_res, y_res = tomeklinks.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def ADASYN(X, y):
    # X, y = load_svmlight_file(file)
    adasyn = imblearn.over_sampling.ADASYN(random_state=None, n_jobs=-1)
    x_res, y_res = adasyn.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def RandomOverSampler(X, y):
    # X, y = load_svmlight_file(file)
    random = imblearn.over_sampling.RandomOverSampler(ratio='auto', random_state=None)
    x_res, y_res = random.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def SMOTE(X, y):
    # X, y = load_svmlight_file(file)
    smote = imblearn.over_sampling.SMOTE(ratio='auto', random_state=None, k_neighbors=5, n_jobs=8)
    x_res, y_res = smote.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def SMOTEBORDER(X, y):
    smote = imblearn.over_sampling.BorderlineSMOTE(sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=8)
    x_res, y_res = smote.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def SMOTENC(X, y):
    # X, y = load_svmlight_file(file)
    smote = imblearn.over_sampling.SMOTENC(n_jobs=-1)
    x_res, y_res = smote.fit_sample(X, y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res


def SMOTEENN(X, y):
    smote = imblearn.combine.SMOTEENN(n_jobs=8)
    x_res, y_res = smote.fit_sample(X, y)
    return x_res, y_res


####
def load_imbalanced_dataset(file):
    X, y = load_svmlight_file(file)
    if globals.config['Imbalanced Datasets']['CondensedNearestNeighbour'] == "True":
        x_res, y_res = CondensedNearestNeighbour(X, y)
    elif globals.config['Imbalanced Datasets']['EditedNearestNeighbours'] == "True":
        x_res, y_res = EditedNearestNeighbours(X, y)
    elif globals.config['Imbalanced Datasets']['RepeatedEditedNearestNeighbour'] == "True":
        x_res, y_res = RepeatedEditedNearestNeighbour(X, y)
    elif globals.config['Imbalanced Datasets']['AllKNN'] == "True":
        x_res, y_res = AllKNN(X, y)
    elif globals.config['Imbalanced Datasets']['InstanceHardnessThreshold'] == "True":
        x_res, y_res = InstanceHardnessThreshold(X, y)
    elif globals.config['Imbalanced Datasets']['NearMiss'] == "True":
        x_res, y_res = NearMiss(X, y)
    elif globals.config['Imbalanced Datasets']['NeighbourhoodCleaningRule'] == "True":
        x_res, y_res = NeighbourhoodCleaningRule(X, y)
    elif globals.config['Imbalanced Datasets']['OneSidedSelection'] == "True":
        x_res, y_res = OneSidedSelection(X, y)
    elif globals.config['Imbalanced Datasets']['RandomUnderSampler'] == "True":
        x_res, y_res = RandomUnderSampler(X, y)
    elif globals.config['Imbalanced Datasets']['TomekLinks'] == "True":
        x_res, y_res = TomekLinks(X, y)
    elif globals.config['Imbalanced Datasets']['ADASYN'] == "True":
        x_res, y_res = ADASYN(X, y)
    elif globals.config['Imbalanced Datasets']['RandomOverSampler'] == "True":
        x_res, y_res = RandomOverSampler(X, y)
    elif globals.config['Imbalanced Datasets']['SMOTE'] == "True":
        x_res, y_res = SMOTE(X, y)
    elif globals.config['Imbalanced Datasets']['SMOTENC'] == "True":
        x_res, y_res = SMOTENC(X, y)
    return x_res, y_res


def Make_Imbalanced_Dataset(X, y):
    globals.logger.debug("Making imbalanced dataset")
    if globals.config['Imbalanced Datasets']['CondensedNearestNeighbour'] == "True":
        x_res, y_res = CondensedNearestNeighbour(X, y)
    elif globals.config['Imbalanced Datasets']['EditedNearestNeighbours'] == "True":
        x_res, y_res = EditedNearestNeighbours(X, y)
    elif globals.config['Imbalanced Datasets']['RepeatedEditedNearestNeighbour'] == "True":
        x_res, y_res = RepeatedEditedNearestNeighbour(X, y)
    elif globals.config['Imbalanced Datasets']['AllKNN'] == "True":
        x_res, y_res = AllKNN(X, y)
    elif globals.config['Imbalanced Datasets']['InstanceHardnessThreshold'] == "True":
        x_res, y_res = InstanceHardnessThreshold(X, y)
    elif globals.config['Imbalanced Datasets']['NearMiss'] == "True":
        x_res, y_res = NearMiss(X, y)
    elif globals.config['Imbalanced Datasets']['NeighbourhoodCleaningRule'] == "True":
        x_res, y_res = NeighbourhoodCleaningRule(X, y)
    elif globals.config['Imbalanced Datasets']['OneSidedSelection'] == "True":
        x_res, y_res = OneSidedSelection(X, y)
    elif globals.config['Imbalanced Datasets']['RandomUnderSampler'] == "True":
        x_res, y_res = RandomUnderSampler(X, y)
    elif globals.config['Imbalanced Datasets']['TomekLinks'] == "True":
        x_res, y_res = TomekLinks(X, y)
    elif globals.config['Imbalanced Datasets']['ADASYN'] == "True":
        x_res, y_res = ADASYN(X, y)
    elif globals.config['Imbalanced Datasets']['RandomOverSampler'] == "True":
        x_res, y_res = RandomOverSampler(X, y)
    elif globals.config['Imbalanced Datasets']['SMOTE'] == "True":
        x_res, y_res = SMOTE(X, y)
    elif globals.config['Imbalanced Datasets']['SMOTEENN'] == "True":
        x_res, y_res = SMOTEENN(X, y)
    elif globals.config['Imbalanced Datasets']['SMOTEBORDERLINE'] == "True":
        x_res, y_res = SMOTEBORDER(X, y)
    elif globals.config['Imbalanced Datasets']['SMOTENC'] == "True":
        x_res, y_res = SMOTENC(X, y)
    return x_res, y_res
