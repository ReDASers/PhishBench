############# README ##################

The platform extract features from emails and URLs, run classifiers, and returns the results using different evaluation metrics.

##### INSTALL
the platform is coded using Python 3.6.4
the code imports multiple modules, and the user will be asked to download them if they're not installed yet.
All the modules can be downloaded using the "pip -install" command
the List of the commands will be found at end of the file.

##### HOW TO RUN THE PROGRAM
all the modules can be run using the python command like so: "python main.py" or "python update.py"

## First run update.py
This will create or update the User_options.py that contains the list of all the features, classifiers, evaluation metrics, and imbalanced dataset options.
The values take either "True" or "False" like so:
```
"Confusion_matrix = True" 
"Cross_validation = False"
```
The entries that are "True" will be executed, the "False" will be ignored. 
It has the option to specify the source of emails and URLs datasets, under  "###### Specify the path for Datasets"
It has the option to choose whether to extract features from emails or from urls: 
```"extract_features_emails = True" 
and "extract_features_urls = True" 
``` 
respectively

Note: In the package that I will upload, the file is already updated, so no need to run it unless you want to add other features or classifiers. If you do, make sure to turn again the values that you previously turned to "False"

## Run main.py
main.py will first run the email feature extraction and url feature extraction. Then it will call for the classifiers in the Classifiers module and then run the evaluation metrics.
The output will show the features being extracted from each file and an error message for each feature if there was an error. The program will not stop if there is an error with a feature extraction.
When the feature extraction is done, the output will show the name and the results of the evaluation metrics for each classifier
IMPORTANT: At this moment the program needs to be run twice: Once for the training data, and once for testing data. The only difference is when specifying the path to the dataset in User_options and change the following options at the end of the file: "Training_data=True" and "Testing_data=False"


##### DESCRIPTION OF MODULES
## Features.py
This module has the definition of all the features that are going to be extracted by the program.
If the user want to add a feature, they should follow the following template: 
```
def feature_name(inputs, list_features, list_time):
    if User_options.feature_name is True:
        start=timeit.timeit()
        code
        list_features["domain_length"]=domain_length
        end=timeit.timeit()
        time=end-start
        list_time["feature_name"]=time
```

if we take the example of a feature that uses 'url' as input then the fucntion will look like this:
```
def url_length(url, list_features, list_time):
    ##global list_features
    if User_options.url_length is True:
        start=timeit.timeit()
        if url=='':
            url_length=0
        else:
            url_length=len(url)
        list_features["url_length"]=url_length
        end=timeit.timeit()
        time=end-start
        list_time["url_length"]=time
```
Notice the test to check if 'url' is empty, then the feature gets 0.
The output of this module are the following:
```
-email_feature_vector_.txt:
-email_features_testing_.txt  (if testing)
-email_features_training_.txt (if training)
-url_feature_vector_.txt
-link_features_testing_.txt (if testing)
-link_features_training_.txt (if training)
```


## Features_Support.py
This module has all the functions that need to run the feature extractions, but are not features per se.
The module is imported into Features.py so any function defined in Features_Support can be called in Feature.py
IMPORTANT: If the user adds a feature in Features.py, then they should also add the following in Feature_Support.py:
```
Features.feature_name(inputs, list_features, list_time)
print("feature_name")
```
This piece of code should be added in one of these different functions in Features_Support.py depending on the nature of the feature: 
```
single_network_features()
single_javascript_features()
single_url_feature()
single_html_features()
single_email_features()
```

## Evaluation_Metrics.py:
The modules have all the code for the evaluation metrics.
If the user wants to add a metric, then they should follow this template:
```
def metric(y_test, y_predict):
    if User_options.metric is True:
        code
        print("metric")
        print(result)
```
and then add the function call in one of the following functions depending on the type of the metric:
```
eval_metrics()
eval_metrics_cluster()
```

## Classifiers.py
This module contains all the classifier that can be run by the platform.
If the user wants to add their own classifier, than they should use the following template:

```
def classifier_name():
    if User_options.classifier_name is True:
        X,y=load_dataset("feature_vector_extract.txt")
        X_test, y_test = load_dataset("feature_vector_extract_test.txt")
        clf = classifier code
        clf.fit(X,y)
        y_predict=clf.predict(X_test)
        print("classifier_name >>>>>>>")
        Evaluation_Metrics.eval_metrics(clf, X, y, y_test, y_predict)
        print("\n")
```

## LIST OF COMMANDS TO INSTALL MODULES
```
pip install -U scikit-learn or pip install sklearn
pip install slimit
pip install tldextract
pip install whois
pip install imblearn
pip install scipy
pip install bs4
pip install cryptography
pip install pandas
pip install matplotlib
pip install nltk
pip install textstat
pip install textblob
pip install dnspython
pip install Cython
pip install ipwhois
pip install python-whois
pip install keras
pip install tensorflow
pip install selenium
pip install html5lib 
```
