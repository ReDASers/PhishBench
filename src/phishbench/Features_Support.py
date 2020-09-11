import ntpath
import os
import os.path

import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

from .utils import phishbench_globals

PAYLOAD_NOT_FOUND = False  # for filtering


def read_corpus(path):
    # assumes a flat directory structure
    files = filter(lambda x: x.endswith('.txt'), os.listdir(path))
    paths = map(lambda x: os.path.join(path, x), files)
    return list(paths)


###########################
########################### Functions for features_url.py


def mean_scaling(Array_Features):
    scaler = preprocessing.StandardScaler().fit(Array_Features)
    # get the mean
    mean_Array_Features = scaler.mean_
    # std_Array=scaler.std_ # does not exist
    # return scaled feature array:
    scaled_Array_Features = scaler.transform(Array_Features)
    return scaled_Array_Features


def min_max_scaling(Array_Features):
    min_max_scaler = preprocessing.MinMaxScaler()
    if np.shape(Array_Features)[0] == 1:
        minmaxScaled_Array_features = min_max_scaler.fit_transform(np.transpose(Array_Features))
        minmaxScaled_Array_features = np.transpose(minmaxScaled_Array_features)
    else:
        minmaxScaled_Array_features = min_max_scaler.fit_transform(Array_Features)
    # get min and max of array features
    min_Array_Features = min_max_scaler.min_
    # max_Array_Features=min_max_scaler.max_  #does not exit
    return minmaxScaled_Array_features


def abs_scaler(Array_Features):
    max_abs_scaler = preprocessing.MaxAbsScaler()
    maxabs_Array_features = max_abs_scaler.fit_transform(Array_Features)
    return maxabs_Array_features


def normalizer(Array_Features):
    Array_Features_normalized = preprocessing.normalize(Array_Features, norm='l2')
    return Array_Features_normalized


def Preprocessing(X):
    # Globals.summary.open(Globals.config["Summary"]["Path"],'w')
    phishbench_globals.summary.write("\n\n###### List of Preprocessing steps:\n")
    # Array_Features=Sparse_Matrix_Features.toarray()
    X_array = X.toarray()
    # X_test_array=X_test.toarray()
    # Center data with the mean and then scale it using the std deviation
    # scaled_Array_Features=preprocessing.scale(Sparse_Matrix_Features)
    # other method that keeps the model for testing
    # if Globals.config["Preprocessing"]["mean_scaling"] == "True":
    #    X_train=mean_scaling(X_train)
    #    X_test=mean_scaling(X_test)
    #    Globals.summary.write("\n Scaling using the mean.\n")
    #    print("Preprocessing: Mean_scaling")
    #    return X_train, X_test
    #    # return the scaler for testing data
    #    # Use min max to scale data because it's robust to very small
    #    # standard deviations of features and preserving zero
    if phishbench_globals.config["Preprocessing"]["min_max_scaling"] == "True":
        X = min_max_scaling(X_array)
        # X_test=min_max_scaling(X_test_array)
        phishbench_globals.summary.write("\n Scaling using the min and max.\n")
        phishbench_globals.logger.info("Preprocessing: min_max_scaling")
        return X
        # use abs value to scale
    # elif Globals.config["Preprocessing"]["abs_scaler"] == "True":
    #    X_train=abs_scaler(X_train)
    #    X_test=abs_scaler(X_test)
    #    Globals.summary.write("\n Scaling using the absolute value.\n")
    #    print("Preprocessing: abs_scaler")
    #    return X_train, X_test
    #    #normalize the data???
    # elif Globals.config["Preprocessing"]["normalize"] == "True":
    #    X_train = normalizer(X_train)
    #    X_test = normalizer(X_test)
    #    Globals.summary.write("\n Normalizing.\n")
    #    print("Preprocessing: Normalizing")
    #    return X_train, X_test
    else:
        return X
    # return scaler, scaled_Array_Features, mean_Array_Features, min_Array_Features #, max_Array_Features


def Cleaning(dict1):
    phishbench_globals.logger.info("Performing Cleaning on dict of len %d", len(dict1))
    count = 0
    for item in dict1:
        # print(item)
        for key in item.keys():
            # print(item[key])
            # if item[key] == "None" or item[key] == "N/A" or item[key] == "Nan" :
            if item[key] in ["None", "N/A", "NaN", None]:
                original = item[key]
                item[key] = -1
                count += 1
                phishbench_globals.logger.debug("Value of {} changed from {} to {}".format(key, original, item[key]))
    phishbench_globals.logger.info("Finished cleaning")


# list_id=list(range(1,len(list_features)+1))
# dict_features=dict(zip(list_id,list_features))
# print(list_features)
# print(list_time)

def Vectorization_Training(list_dict_features_train):
    vec = DictVectorizer()
    vec.fit(list_dict_features_train)
    sparse_matrix_features_train = vec.transform(list_dict_features_train)
    return sparse_matrix_features_train, vec


# sys.setdefaultencoding('utf-8')

# get filename from path
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
