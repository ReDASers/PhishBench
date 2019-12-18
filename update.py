import os
import sys
import Features
import Classifiers
import Imbalanced_Dataset
import Evaluation_Metrics
import inspect
import configparser
#from Classifiers_test import load_dataset

def config(list_Features, list_Classifiers, list_Imbalanced_dataset, list_Evaluation_metrics):
	config = configparser.ConfigParser()

	config['Features'] = {}
	config['Email_Features']={}
	#C_Features=config['Features']
	C_Email_Features=config['Email_Features']
	for feature in list_Features:
		if feature.startswith("Email_"):
			C_Email_Features[feature.replace('Email_','')]="True"
		#C_Features[feature]="True"

	config['HTML_Features']={}
	C_HTML_Features=config['HTML_Features']
	for feature in list_Features:
		if feature.startswith("HTML_"):
			C_HTML_Features[feature.replace('HTML_','')]="True"

	config['URL_Features']={}
	C_URL_Features=config['URL_Features']	
	for feature in list_Features:
		if feature.startswith("URL_"):
			C_URL_Features[feature.replace('URL_','')]="True"


	config['Network_Features']={}
	C_Network_Features=config['Network_Features']
	for feature in list_Features:
		if feature.startswith("Network_"):
			C_Network_Features[feature.replace('Network_','')]="True"


	config['Javascript_Features']={}
	C_Javascript_Features=config['Javascript_Features']
	for feature in list_Features:
		if feature.startswith("Javascript_"):
			C_Javascript_Features[feature.replace('Javascript_','')]="True"

	config['Classifiers']={}
	C_Classifiers=config['Classifiers']
	for classifier in list_Classifiers:
		C_Classifiers[classifier]="True"

	config['Imbalanced Datasets'] = {}
	C_Imbalanced=config['Imbalanced Datasets']
	C_Imbalanced["load_imbalanced_dataset"]="False"
	for imbalanced in list_Imbalanced_dataset:
		C_Imbalanced[imbalanced]="True"

	config['Evaluation Metrics']={}
	C_Metrics=config['Evaluation Metrics']
	for metric in list_Evaluation_metrics:
		C_Metrics[metric]="True"

	config['Preprocessing']={}
	C_Preprocessing=config['Preprocessing']
	#C_Preprocessing['mean_scaling']= "True"
	C_Preprocessing['mix_max_scaling']= "True"
	#C_Preprocessing['abs_scaler']= "True"
	#C_Preprocessing['normalize']= "True"
	

	config["Feature Selection"]={}
	C_selection=config["Feature Selection"]
	C_selection["Select Best Features"]="True"
	C_selection["Number of Best Features"]="80"
	C_selection["Feature Ranking Only"]="False"
	C_selection["Recursive Feature Elimination"]="False"
	C_selection["Information Gain"]="True"
	C_selection["Gini"]="False"
	C_selection["Chi-2"]="False"

	config['Dataset Path']={}
	C_Dataset=config['Dataset Path']
	C_Dataset["path_legitimate_training"]="Dataset_all/Dataset_legit_urls"
	C_Dataset["path_phishing_training"]="Dataset_all/Dataset_phish_urls"
	C_Dataset["path_legitimate_testing"]="Dataset_all/Dataset_legit_urls"
	C_Dataset["path_phishing_testing"]="Dataset_all/Dataset_legit_urls"

	config['Email or URL feature Extraction']={}
	C_email_url=config['Email or URL feature Extraction']
	C_email_url["extract_features_emails"]="False"
	C_email_url["extract_features_urls"]="True"

	config['Extraction']={}
	C_extraction=config['Extraction']
	C_extraction["Feature Extraction"]="True"
	C_extraction["Training Dataset"]="True"
	C_extraction["Testing Dataset"]="True"

	config['Features Format']={}
	C_features_format=config['Features Format']
	C_features_format["Pikle"]="True"
	C_features_format["Svmlight format"]="True"


	config['Classification']={}
	C_classification=config['Classification']
	C_classification["Running the Classifiers"]="True"
	C_classification["Save Models"]="True"

	config["Summary"]={}
	C_summary=config["Summary"]
	C_summary["Path"]="summary.txt"

	with open('Config_file.ini', 'w') as configfile:
		config.write(configfile)


def update_list():
	list_Features=[]
	list_Classifiers=[]
	list_Evaluation_metrics=[]
	list_Imbalanced_dataset=[]
	for a in dir(Features):
		element=getattr(Features, a)
		if inspect.isfunction(element):
			list_Features.append(a)

	for a in dir(Classifiers):
		element=getattr(Classifiers, a)
		if inspect.isfunction(element):
			list_Classifiers.append(a)

	for a in dir(Imbalanced_Dataset):
		element=getattr(Imbalanced_Dataset, a)
		if inspect.isfunction(element):
			list_Imbalanced_dataset.append(a)

	for a in dir(Evaluation_Metrics):
		element=getattr(Evaluation_Metrics, a)
		if inspect.isfunction(element):
			list_Evaluation_metrics.append(a)

	return list_Features, list_Classifiers, list_Imbalanced_dataset, list_Evaluation_metrics


if __name__ == "__main__":
    # execute only if run as a script
    list_Features, list_Classifiers, list_Imbalanced_dataset, list_Evaluation_metrics = update_list()
    #update_file(list_Features, list_Classifiers, list_Imbalanced_dataset, list_Evaluation_metrics)
    config(list_Features, list_Classifiers, list_Imbalanced_dataset, list_Evaluation_metrics)
