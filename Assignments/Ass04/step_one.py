# COS 470/570 NLP spring 2023
# Assignment 04
# Sean Fletcher

# imports
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from post_parser_record import PostParserRecord
from sklearn.metrics.pairwise import cosine_similarity
import re
# end of imports


# function definitions
def html_tag_remover(input_string):
    """
    this function uses BeautifulSoup to remove the HTML element tags from a string
    :param input_string:
    :return: text: a string without the HTML element tags
    """
    soup = BeautifulSoup(input_string, "html.parser")
    text = soup.get_text()
    return text


def get_embedding(model, text):
    # This method takes in the model and the input text
    # and returns a 768-dimensional embedding of the text
    embedding = model.encode([text])
    return embedding


def dict_to_tsv_file(input_dictionary, file_name):
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for key, value in input_dictionary.items():
            writer.writerow([key, value])


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

    return embeddings_dict, wrong_ids


def make_large_ppr_embedding_dict(post_parser_record, model):
    embeddings_dict = {}
    wrong_ids = []

    for question_id in post_parser_record.map_questions:
        try:  # this try-except block catches bogus question ids that appear in the ppr
            question_obj = post_parser_record.map_questions[question_id]
        except KeyError:
            wrong_ids.append(question_id)
            continue

        question_title = question_obj.title
        question_body = question_obj.body
        question_text = question_title + question_body
        clean_question_text = html_tag_remover(question_text)
        clean_question_embedding = get_embedding(model, clean_question_text)

        embeddings_dict[question_id] = clean_question_embedding

    return embeddings_dict, wrong_ids


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


def get_p_at_1_score_average(similar_question_dict, top_100_dict):
    number_of_questions = 0
    number_of_p_at_1s = 0
    for key in top_100_dict:
        number_of_questions += 1
        top_pick = top_100_dict[key][0][1]
        similar_question_list = similar_question_dict[key]  # this list is of length 1 or 2
        if top_pick in similar_question_list:
            number_of_p_at_1s += 1
    return number_of_p_at_1s / number_of_questions


def get_mmr_score_average(similar_questions_dict, top_100_dict):
    mmr_scores = []
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
                    mmr_scores.append(1 / current_rank)
                else:
                    mmr_scores.append(0)
            else:
                if similar_question_id01 == tupe[0]:
                    mmr_scores.append(1 / current_rank)
                else:
                    mmr_scores.append(0)
            current_rank += 1
    return sum(mmr_scores)/len(mmr_scores)
# end of function definitions


# function calls
the_distilbert_model = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')
parsed_law_posts = PostParserRecord("Posts_law.xml")

target_question_embedding_dict00, list_of_error_ids_01 = make_target_question_embedding_dict(
    "duplicate_questions.tsv",
    parsed_law_posts,
    the_distilbert_model)
dict_to_tsv_file(target_question_embedding_dict00, "target_question_IDs_and_embeddings")

ppr_question_embedding_dict00, list_of_error_ids_02 = make_large_ppr_embedding_dict(
    parsed_law_posts,
    the_distilbert_model)
dict_to_tsv_file(ppr_question_embedding_dict00, "large_ppr_question_IDs_and_embeddings")

target_question_embedding_dict = tsv_embeddings_file_to_dict("target_question_IDs_and_embeddings")
ppr_question_embedding_dict = tsv_embeddings_file_to_dict("large_ppr_question_IDs_and_embeddings")

top_100 = get_top_100_similar_questions(target_question_embedding_dict, ppr_question_embedding_dict)
dict_to_tsv_file(top_100, "top_100_list_for_each_target_question")

hundo_dict = tsv_top_100_file_to_dict("top_100_list_for_each_target_question")
similar_q_dict = make_similar_questions_dict("duplicate_questions.tsv")

print("*****")
print("P@1 score average:")
print(get_p_at_1_score_average(similar_q_dict, hundo_dict))
print("*****")
print("*****")
print("MMR score average:")
print(get_mmr_score_average(similar_q_dict, hundo_dict))
print("*****")
# end of function calls
