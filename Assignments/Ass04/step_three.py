# COS 470/570 NLP spring 2023
# Assignment 04
# Sean Fletcher

from bs4 import BeautifulSoup
import csv
import re
from scipy import stats
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from post_parser_record import PostParserRecord
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def html_tag_remover(input_string):
    """
    this function uses BeautifulSoup to remove the HTML element tags from a string
    :param input_string:
    :return: text: a string without the HTML element tags
    """
    soup = BeautifulSoup(input_string, "html.parser")
    text = soup.get_text()
    return text


def dict_to_tsv_file(input_dictionary, file_name):
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for key, value in input_dictionary.items():
            writer.writerow([key, value])


def get_embedding(model, text):
    # This method takes in the model and the input text
    # and returns a 768-dimensional embedding of the text
    embedding = model.encode([text])
    return embedding


def tsv_top_100_file_to_dict(file_name):
    top_100_dict = {}

    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            target_q_id = int(row[0])

            pattern = r'\((\d+),\s*array\(\[\[(\d+\.\d+)\]\]\)\)'
            matches = re.findall(pattern, row[1])
            list_of_matches = []
            for match in matches:
                list_of_matches.append((int(match[0]), float(match[1])))
            top_100_dict[target_q_id] = list_of_matches
    return top_100_dict


def make_similar_questions_dict(tsv_file):
    similar_qs_dict = {}
    with open(tsv_file, 'r') as t_s_v:

        reader = csv.reader(t_s_v, delimiter='\t')

        for row in reader:  # each row has a list of two or three question ID numbers

            test_question_id = int(row[0])
            similar_question_ids = [int(row[1])]
            try:
                similar_id_02 = int(row[2])
                similar_question_ids.append(similar_id_02)
            except IndexError:
                pass
            similar_qs_dict[test_question_id] = similar_question_ids
    return similar_qs_dict


def make_target_question_embedding_dict(tsv_file, post_parser_record, model):
    embeddings_dict = {}
    wrong_ids = []

    with open(tsv_file, 'r') as t_s_v:

        reader = csv.reader(t_s_v, delimiter='\t')

        for row in reader:
            # each row has a list of two or three question ID numbers,
            # this function is only concerned with the first.

            target_question_id = int(row[0])

            try:
                target_question = post_parser_record.map_questions[target_question_id]
            except KeyError:
                wrong_ids.append(target_question_id)
                continue

            target_title = target_question.title
            target_body = target_question.body
            the_question = target_title + target_body
            clean_target_text = html_tag_remover(the_question)
            target_embedding = get_embedding(model, clean_target_text)
            embeddings_dict[target_question_id] = target_embedding

    return embeddings_dict


def get_top_100_similar_questions(target_q_embedding_dict, ppr_q_embedding_dict):
    """

    :param target_q_embedding_dict:
    :param ppr_q_embedding_dict:
    :return: a dictionary with the target-question-id as the key and the list of tuples
             length == 100; where tuple[0] is a question id and tuple[1] is a cosine similarity
             it is the top 100 cosine similarities for the target-question
    """
    similarities = {}
    for target_key, target_embedding in target_q_embedding_dict.items():
        similarity_list = []
        for ppr_key, ppr_embedding in ppr_q_embedding_dict.items():

            if target_key == ppr_key:
                continue
            else:
                cosine_sim = cosine_similarity(target_embedding, ppr_embedding)

            # converting the embeddings to numpy arrays for computation
            # t_array = np.array(target_embedding)
            # ppr_array = np.array(ppr_embedding)

            # compute cosine similarity
            # cosine_sim = cosine_similarity(t_array, ppr_array)

            # cosine_sim = 1 - spatial.distance.cosine(t_array, ppr_array)
            # cosine_sim = np.dot(t_array, ppr_array) / (norm(t_array) * norm(ppr_array))

            # building a top 100 list
            if len(similarity_list) < 100:
                similarity_list.append((ppr_key, cosine_sim))
                similarity_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)

            else:
                if cosine_sim > similarity_list[-1][1]:
                    similarity_list.append((ppr_key, cosine_sim))
                    similarity_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)
                    similarity_list = similarity_list[:100]

        similarities[target_key] = similarity_list
    return similarities


def tsv_embeddings_file_to_dict(file_name):
    dict_to_return = {}
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            key = int(row[0])
            value = row[1]
            value = value.replace('\n', '')
            value_list = value.replace('[', '').replace(']', '').replace('\n', '').split(' ')

            # convert the string values to float values
            float_values = []
            for value in value_list:
                if value != "":
                    float_value = float(value)
                    float_values.append(float_value)
            numpy_array = np.array(float_values)
            numpy_array = numpy_array.reshape(1, -1)

            dict_to_return[key] = numpy_array
    return dict_to_return


def get_mmr_scores(similar_questions_dict, top_100_dict):
    mmr_scores_0 = []
    for question_id, list_of_tuples in top_100_dict.items():
        current_rank = 1
        if len(similar_questions_dict[question_id]) == 1:
            similar_question_id01 = similar_questions_dict[question_id][0]
            similar_question_id02 = 0
        elif len(similar_questions_dict[question_id]) == 2:
            similar_question_id01 = similar_questions_dict[question_id][0]
            similar_question_id02 = similar_questions_dict[question_id][1]
        else:
            print("something went wrong here... this shouldn't print... line 221")
            similar_question_id01 = 0
            similar_question_id02 = 0
        for tupe in list_of_tuples:
            if similar_question_id02 != 0:
                if similar_question_id01 == tupe[0] or similar_question_id02 == tupe[0]:
                    mmr_scores_0.append(1 / current_rank)
                else:
                    mmr_scores_0.append(0)
            else:
                if similar_question_id01 == tupe[0]:
                    mmr_scores_0.append(1 / current_rank)
                else:
                    mmr_scores_0.append(0)
            current_rank += 1
    return mmr_scores_0


the_distilbert_model = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')
parsed_law_posts = PostParserRecord("Posts_law.xml")
fine_tuned_model = SentenceTransformer("./law_post_bert_model")


# getting the embeddings and top 100 matching embeddings based on cosine similarities for the
# base model
base_model_test_questions_embeddings_dict = make_target_question_embedding_dict(
    "ids_test_set.tsv", parsed_law_posts, the_distilbert_model)
dict_to_tsv_file(base_model_test_questions_embeddings_dict, "base_model_test_question_IDs_and_embeddings")
base_model_test_questions_embeddings_dict = tsv_embeddings_file_to_dict("base_model_test_question_IDs_and_embeddings")
base_model_large_ppr_embeddings_dict = tsv_embeddings_file_to_dict("large_ppr_question_IDs_and_embeddings")
base_model_top_100_test_dict = get_top_100_similar_questions(
    base_model_test_questions_embeddings_dict, base_model_large_ppr_embeddings_dict)
dict_to_tsv_file(base_model_top_100_test_dict, "base_model_top_100_list_for_each_test_question")


# getting the top 100 similar question ids and embeddings by the base model and the
# fine-tuned model to compare MRR scores in order to find the p value
base_model_hundo_test_dict = tsv_top_100_file_to_dict("base_model_top_100_list_for_each_test_question")
fine_tuned_hundo_test_dict = tsv_top_100_file_to_dict("fine_tuned_top_100_list_for_each_test_question")
similar_q_dict = make_similar_questions_dict("duplicate_questions.tsv")


# MRR for sample 1
mmr_1 = get_mmr_scores(similar_q_dict, base_model_hundo_test_dict)

# MMR for sample 2
mmr_2 = get_mmr_scores(similar_q_dict, fine_tuned_hundo_test_dict)

# Sample sizes
n1 = len(mmr_1)
n2 = len(mmr_2)

# Sample variances
var1 = stats.tvar(mmr_1)
var2 = stats.tvar(mmr_2)

# Perform two-sample t-test with MMR
t_stat, p_value = stats.ttest_ind(mmr_1, mmr_2, equal_var=False)

# Print the results
print("t-statistic:", t_stat)
print("p-value:", p_value)
