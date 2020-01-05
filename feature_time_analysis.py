
import argparse
import re

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('--time_files', type=str, required=True, nargs=2,
                    help='File that contains all the feature extraction times.')

args = parser.parse_args()


prog = re.compile("('[a-zA-Z0-9_\-\. ]*':-?[0-9\.0-9]*e-[0-9]+)|('[a-zA-Z0-9_\-\. ]*':-?[0-9\.0-9]*)")

def read_feature_time(input_file):
    dict = {}
    total_urls = 0
    print(input_file)
    with open(input_file, "r") as file_descriptor:
        for line in file_descriptor:
            line = line.strip().rstrip()
            if line != '\n' and line.startswith('URL:') == False:
                total_urls = total_urls+1
                tuple_regex_feature = prog.findall(line)
                split_feature = [*map(''.join,tuple_regex_feature)]
                for feature in split_feature:
                    item = feature.split(':')
                    key = (item[0])[1:-1]
                    value = item[1]
                    if key in dict:
                        dict[key] = dict[key] + float(item[1])
                    else:
                        dict[key] = float(item[1])
    return dict,total_urls

def main():
    print("Loading input file")
    time_dict, num = read_feature_time(args.time_files[0])
    time_dict2, num2 = read_feature_time(args.time_files[1])
    for key, value in time_dict.items():
        print("{}:{}".format(key, (value+time_dict2[key])/(num+num2)))

if __name__ == "__main__":
    main()
