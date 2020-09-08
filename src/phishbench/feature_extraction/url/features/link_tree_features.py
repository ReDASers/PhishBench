import statistics

from bs4 import BeautifulSoup

from ...reflection import FeatureType, register_feature
from ....input import URLData


@register_feature(FeatureType.URL_WEBSITE, 'link_tree_features')
def link_tree_features(url: URLData):
    domain = url.final_url.split("//")[-1].split("/")[0]
    link_features = img_features = video_features = a_features = meta_features = script_features = [[0, 0, 0],
                                                                                                    [0, 0, 0],
                                                                                                    [0, 0, 0],
                                                                                                    [0, 0, 0],
                                                                                                    [0, 0, 0]]
    soup = BeautifulSoup(url.downloaded_website, 'html5lib')

    # get links from content
    link_link = tree_get_links(soup, 'link', 'href', '')
    img_link = tree_get_links(soup, 'img', 'src', '')
    video_link = tree_get_links(soup, 'video', 'src', '')
    a_link = tree_get_links(soup, 'a', 'src', '')
    a_link += tree_get_links(soup, 'a', 'href', '')
    meta_link = tree_get_links(soup, 'meta', 'content', '/')
    script_link = tree_get_links(soup, 'script', 'src', '')

    # extract features: size, mean, standard deviation
    link_features = extract_tree_features(link_link, domain)
    img_features = extract_tree_features(img_link, domain)
    video_features = extract_tree_features(video_link, domain)
    a_features = extract_tree_features(a_link, domain)
    meta_features = extract_tree_features(meta_link, domain)
    script_features = extract_tree_features(script_link, domain)

    features = {}
    add_features(features, link_features, 'link')
    add_features(features, img_features, 'img')
    add_features(features, video_features, 'video')
    add_features(features, a_features, 'a')
    add_features(features, meta_features, 'meta')
    add_features(features, script_features, 'script')
    return features


def tree_get_links(soup, tag, source, identifier):
    links = []
    for link in soup.findAll(tag):
        content = link.get(source)
        if content is not None:
            if identifier in content:
                links.append(content)
    return links


# Get size, mean, SD for a set of links
def extract_tree_features(links, domain):
    # TODO could use a input file for the list, not hardcoded!
    social_list = ['google.com', 'facebook.com', 'twitter.com', 'pinterest.com', 'instagram.com']
    features = []
    t1 = t2 = t3 = t4 = t5 = []
    for link in links:
        link_domain = link.split("//")[-1].split("/")[0]
        if domain in link:
            t1.append(link)
        elif link_domain in social_list:
            t2.append(link)
        elif "https:" == link[:6]:
            t3.append(link)
        elif "http:" == link[:5]:
            t4.append(link)
        else:
            t5.append(link)
    features.append(extract_by_type(t1))
    features.append(extract_by_type(t2))
    features.append(extract_by_type(t3))
    features.append(extract_by_type(t4))
    features.append(extract_by_type(t5))
    return features


def extract_by_type(link_list):
    block_size = len(link_list)
    block_mean = 0
    block_std = 0
    links_length = []
    for link in link_list:
        links_length.append(len(link))
    if len(links_length) > 0:
        block_mean = round(statistics.mean(links_length), 2)
    if len(links_length) > 1:
        block_std = round(statistics.stdev(links_length), 2)
    return [block_size, block_mean, block_std]


def add_features(list_features, features, tag):
    # features = [[3],[3],[3],[3],[3]], ..., [[],[],[],[],[]]
    # T1 - T5: 5 * 3
    name_list = ['size', 'mean', 'SD']
    for i, set_f in enumerate(features):
        # size mean, SD
        name_1 = "LTree_feature_" + str(tag) + "_T" + str(i) + "_"
        for j, value in enumerate(set_f):
            feature_name = name_1 + name_list[j]
            list_features[feature_name] = value
