import csv
import json
import os
import markdown
import requests
from bs4 import BeautifulSoup
import yake
from sentence_transformers import SentenceTransformer, InputExample, losses, models
import torch
from torch.utils.data import Dataset
import re


def prepare_data_for_fine_tuning(text01, text02, label):
    # this function formats the data, so it can be used for fine tuning roberta
    data_00 = InputExample(texts=[text01, text02], label=label)
    return data_00


def make_elastic_call_for_single_article(article_id, target_dir):
    url_start = 'https://guacamole.univ-avignon.fr/dblp1/_search?q=_id:'

    if type(article_id) != str:
        article_id = str(article_id)
    full_url = url_start + article_id

    user, password = 'inex', 'qatc2011'

    result = requests.get(full_url, auth=(user, password)).content.decode("utf-8")
    obj = json.loads(result)

    with open(target_dir + article_id + ".json", "w") as file:
        json.dump(obj, file)


def split_the_qrel_file(qrel_file_path, name_of_new_training_qrel_file,
                        name_of_new_validation_qrel_file, name_of_new_test_qrel_file):
    qrel_file = open(qrel_file_path, "r")

    # Initialize a dictionary
    the_qrel_dict = {}

    # Parse the qrel file line by line and turn it into a dictionary
    for line in qrel_file:

        # Split the line into its components (query ID, document ID, and relevance score)
        components = line.split()
        query_id = components[0]
        # components[1] is just a bunch of zeros. ChatGPT won't tell me why this is so... what a punk
        document_id = components[2]
        relevance_score = int(components[3])

        if query_id not in the_qrel_dict:
            # document IDs are the keys because they are unique, the others have multiple duplicates
            the_qrel_dict[document_id] = (query_id, relevance_score)
        else:
            print("HOLD UP!!! The SAME document ID is showing up TWICE, WHY GOD WHY?!?!?!")

    # Initialize two new dictionaries
    training_qrel_dict = {}
    validation_and_test_qrel_dict = {}
    validation_qrel_dict = {}
    test_qrel_dict = {}

    # Loop through the original dictionary using enumerate()
    # to keep track of the index of each key-value pair.
    for i, (key, value) in enumerate(the_qrel_dict.items()):

        # Every 5th key-value pair, add it to the validation_and_test dict
        # (20%, which I split into 10% and 10% for validation and test later)
        if i % 5 == 0:
            validation_and_test_qrel_dict[key] = value
        # Otherwise, add it to the training dict (80% for training)
        else:
            training_qrel_dict[key] = value

    # this loop is for splitting the 20% in the Val_and_Test dict into separate validation and test sets
    for i, (key, value) in enumerate(validation_and_test_qrel_dict.items()):

        # this splits the 20% Val_and_Test dict from above in half,
        # 10% for validation, 10% for test.
        if i % 2 == 0:
            validation_qrel_dict[key] = value
        # Otherwise, add it to the test dict
        else:
            test_qrel_dict[key] = value

    # writing the training qrel dict into a qrel file
    with open(name_of_new_training_qrel_file, "w") as qrel_file:
        for doc_id, (q_id, score) in training_qrel_dict.items():
            qrel_file.write("{} 0 {} {}\n".format(q_id, doc_id, score))

    # writing the validation qrel dict into a qrel file
    with open(name_of_new_validation_qrel_file, "w") as qrel_file:
        for doc_id, (q_id, score) in validation_qrel_dict.items():
            qrel_file.write("{} 0 {} {}\n".format(q_id, doc_id, score))

    # writing the test qrel dict into a qrel file
    with open(name_of_new_test_qrel_file, "w") as qrel_file:
        for doc_id, (q_id, score) in test_qrel_dict.items():
            qrel_file.write("{} 0 {} {}\n".format(q_id, doc_id, score))


def read_json(filename: str):
    # reads one json file
    with open(filename) as f_in:
        return json.load(f_in)


def read_all_jsons(target_dir):
    # takes in teh directory of topics and return dictionary of topics with id corresponding to qrel files
    dict_top_res = {}
    for file in os.listdir(target_dir):
        temp_dic = read_json(target_dir + file)
        hits = temp_dic['hits']['hits']
        temp_dic_result = {}
        for hit in hits:
            source = hit['_source']
            paper_id = source['id']
            title = source['title']
            abstract = source['abstract']
            temp_dic_result[paper_id] = (title, abstract)

        query_id = file.split(".")[0]
        # if len(temp_dic_result) < 2000:
        #     print(query_id + "\t" + str(len(temp_dic_result)))
        dict_top_res[query_id] = temp_dic_result
    return dict_top_res


def read_all_jsons_for_baseline(target_dir):
    # takes in teh directory of topics and return dictionary of topics with id corresponding to qrel files
    dict_top_res = {}
    for file in os.listdir(target_dir):
        temp_dic = read_json(target_dir + file)
        hits = temp_dic['hits']['hits']
        temp_dic_result = {}
        for hit in hits:
            source = hit['_source']
            score = hit['_score']
            paper_id = source['id']
            title = source['title']
            abstract = source['abstract']
            temp_dic_result[paper_id] = (title, abstract, score)

        query_id = file.split(".")[0]
        # if len(temp_dic_result) < 2000:
        #     print(query_id + "\t" + str(len(temp_dic_result)))
        dict_top_res[query_id] = temp_dic_result
    return dict_top_res


def read_topic_file(topic_filepath):
    # a method used to read the topic file for this year of the lab; to be passed to BERT/PyTerrier methods
    result = {}
    with open(topic_filepath, "r") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        pre_qid = ""
        counter = 1
        for line in reader:
            query_to_es = line[-1]
            original_query = query_to_es.split("q=")[1][1:-1]
            topic_text = line[1]
            q_id = line[0]
            if q_id == pre_qid:
                counter += 1
            else:
                counter = 1
            pre_qid = q_id
            result[q_id + "_" + str(counter)] = (original_query, topic_text)
    return result


def extract_keywords(directory_path, size):
    # generating new queries using YAKE to include more keywords in the query
    result = {}
    language = "en"
    max_ngram_size = 1
    deduplication_threshold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = size
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size,
                                                dedupLim=deduplication_threshold,
                                                dedupFunc=deduplication_algo, windowsSize=windowSize,
                                                top=numOfKeywords, features=None)
    for file in os.listdir(directory_path):
        f = open(directory_path + file, 'r', encoding="utf-8")
        htmlmarkdown = markdown.markdown(f.read())
        # print(htmlmarkdown)
        cleantext = BeautifulSoup(htmlmarkdown, "lxml").text
        # print(cleantext)
        keywords = custom_kw_extractor.extract_keywords(cleantext)
        n = 0  # N. . .
        keywords = [x[n] for x in keywords]
        string = " ".join(keywords)
        topic_id = file.split("topic")[1].split(".")[0]
        result[topic_id] = string
    return result


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # process sample here
        return sample


def get_the_query_ids_from_the_qrel_file(file_path):
    qrel_file = open(file_path, "r")

    query_id_list = []

    # Parse the qrel file line by line and turn it into a dictionary
    for line in qrel_file:
        # Split the line into its components (query ID, document ID, and relevance score)
        components = line.split()
        if components[0] not in query_id_list:
            query_id_list.append(components[0])

    return query_id_list


def get_the_query_search_term_from_the_simpletext_csv_file(file_path):
    dict_of_query_terms = {
        'G01.1': '',
        'G01.2': '',
        'G02.1': '',
        'G02.2': '',
        'G03.1': '',
        'G03.2': '',
        'G04.1': '',
        'G04.2': '',
        'G04.3': '',
        'G05.1': '',
        'G05.2': '',
        'G06.1': '',
        'G06.2': '',
        'G07.1': '',
        'G07.2': '',
        'G08.1': '',
        'G08.2': '',
        'G09.1': '',
        'G10.1': '',
        'G10.2': '',
        'G11.1': '',
        'G12.1': '',
        'G13.1': '',
        'G13.2': '',
        'G14.1': '',
        'G14.2': '',
        'G15.1': '',
        'G15.2': '',
        'G15.3': '',
        'G15.4': '',
        'G16.1': '',
        'G16.2': '',
        'G16.3': '',
        'G16.4': '',
        'G17.1': '',
        'G17.2': '',
        'G17.3': '',
        'G17.4': '',
        'G18.1': '',
        'G18.2': '',
        'G18.3': '',
        'G18.4': '',
        'G19.1': '',
        'G19.2': '',
        'G19.3': '',
        'G20.1': '',
        'G20.2': '',
    }
    simple_text_query_search_term_list = []

    combined_dict = {}

    counter = 0
    with open(file_path, 'r') as c_s_v:

        reader = csv.reader(c_s_v, delimiter=';')
        pattern = r'search\?q=(.*)'  # this will target the query terms in the url

        for row in reader:  # each row has 6 strings: topic_id;topic_text;topic_url;query_id;query_text;abstract_url

            if counter == 0:  # this is if-else block is to skip the first row of the CSV file
                counter += 1
                continue
            else:
                if row[0][0] == 'G':  # so I only get the query search terms for The Guardian articles
                    match = re.search(pattern, row[5])
                    if match:
                        query_term = match.group(1)
                        if query_term[0] == '"':  # this block removes any quotation marks around the query term
                            query_term = query_term.replace('"', '')
                        simple_text_query_search_term_list.append(query_term)

    query_ids = list(dict_of_query_terms.keys())
    for i in range(len(simple_text_query_search_term_list)):
        combined_dict[query_ids[i-1]] = simple_text_query_search_term_list[i-1]

    return combined_dict


def long_keyword_splitter(n_gram_size, keyword_string):
    new_list_of_n_gram_sized_keywords = []
    key_word_string_to_list = keyword_string.split()
    count = len(key_word_string_to_list)
    if count <= n_gram_size:
        new_list_of_n_gram_sized_keywords.append(keyword_string)
    else:
        for place in range(count):
            i = place - 1
            j = n_gram_size + place - 1
            if count < j:
                continue
            new_keyword = ' '.join(key_word_string_to_list[i:j])
            if new_keyword != '':
                new_list_of_n_gram_sized_keywords.append(new_keyword)
            else:
                continue
    return new_list_of_n_gram_sized_keywords


def compare_keyword_lists(kw_lst_01, kw_lst_02):
    total_hits = 0
    for kw in kw_lst_02:
        if kw in kw_lst_01:
            total_hits += 1
    if total_hits == 0:
        return 0.0
    else:
        return total_hits/len(kw_lst_02)


def top_keyword_comparisons(query_article_kw_dict, sci_abstract_dict_from_json, number_of_results, yake_obj_kw_extractor):
    dict_to_return = {}  # {'G01.1': [(doc_id, best_kw_score), (doc_id, 2nd_best_kw_score)]}
    for key, q_dict in sci_abstract_dict_from_json.items():

        reformatted_key = key.replace("_", ".")  # gets the query id in G01.1 format instead of G01_1
        if reformatted_key[0] == 'G':
            dict_to_return[reformatted_key] = []  # this list will hold the top n (number_of_results) kw comparison results
        else:
            continue

        q_kw_lst = query_article_kw_dict[reformatted_key]
        for doc_id, sci_content_tuple in q_dict.items():  # sci_content_tuple: ('title', 'abstract_body')
            sci_article_title = sci_content_tuple[0]
            sci_article_body = sci_content_tuple[1]
            text = sci_article_title + sci_article_body
            keyword_tupels = yake_obj_kw_extractor.extract_keywords(text)  # using YAKE to get the keywords
            list_of_sci_article_kws = []
            for tupe in keyword_tupels:
                t_k = tupe[0]  # gets the keyword from the YAKE tuple
                t_k_lst = long_keyword_splitter(2, t_k)  # splits the keyword into ngrams size 2, if len > 3
                list_of_sci_article_kws += t_k_lst
            kw_comp_score = compare_keyword_lists(q_kw_lst, list_of_sci_article_kws)  # comparing the keyword_lists against eachother
            if len(dict_to_return[reformatted_key]) < number_of_results:
                dict_to_return[reformatted_key].append((doc_id, kw_comp_score))
                # sort the dict each time in descending order each time
                dict_to_return[reformatted_key] = sorted(dict_to_return[reformatted_key], key=lambda x: -x[1])
            else:
                the_list = dict_to_return[reformatted_key]
                the_lowest_score = the_list[-1]
                if kw_comp_score > dict_to_return[reformatted_key][-1][1]:  # if the comp_score is bigger than the smallest comp score
                    dict_to_return[reformatted_key].append((doc_id, kw_comp_score))
                    dict_to_return[reformatted_key] = sorted(dict_to_return[reformatted_key], key=lambda x: -x[1])
                    dict_to_return[reformatted_key] = dict_to_return[reformatted_key][:number_of_results]
    return dict_to_return


def dict_to_txt(input_dict, file_name):
    with open(file_name, 'w') as f:
        for key, value in input_dict.items():
            for i, (doc_id, comp_score) in enumerate(value):
                f.write(f"{key} Q0 {doc_id} {i + 1} {comp_score} YAKE\n")

