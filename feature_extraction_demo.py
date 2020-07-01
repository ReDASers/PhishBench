from phishbench.feature_extraction.email.reflection import extract_features_emails
import phishbench.input as Input
import phishbench.utils.Globals as Globals

import pandas as pd

if __name__ == "__main__":
    Globals.setup_globals()
    folder =r"E:\Repos\PhishBench\repo\SampleDataset"
    bodies, headers, emails, files = Input.read_dataset_email(folder)
    print("{} emails loaded".format(len(bodies)))
    feature_dict_list, time_dict_list = extract_features_emails(bodies, headers)
    df = pd.DataFrame(feature_dict_list, index=files)
    print(df)


