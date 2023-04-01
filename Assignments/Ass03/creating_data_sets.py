# imports
import csv
from gensim.models import FastText
from post_parser_record import PostParserRecord
from Ass_03 import html_tag_remover, get_sentence_embedding
# end of imports


# function definitions
# function definitions section: get the IDs
def get_question_ids(post_parser):
    list_of_question_ids = []
    for q_id in post_parser.map_questions:
        list_of_question_ids.append(q_id)
    return list_of_question_ids


def parse_the_tsv_file_ids_only(tsv_file):
    # this method returns a list of the question ids and a list of question ids of questions
    # that are "similar" to the first question. These IDs are found in the TSV file
    # provided by Professor Mansouri

    # this method is used to help create positive and negative examples for training a feed
    # forward neural network for prediction.

    list_of_questions = []
    list_of_similar_questions = []
    list_of_bad_ids = [57017, 74161, 73947, 74939, 78262, 83650, 84555]

    with open(tsv_file, 'r') as tsvfile:
        # Create a reader object
        reader = csv.reader(tsvfile, delimiter='\t')
        # Loop over each row in the file
        for row in reader:  # each row has a list of two or three question ID numbers

            question_id = int(row[0])
            if question_id in list_of_bad_ids:
                continue
            else:
                list_of_questions.append(question_id)

            similar_question_id = int(row[1])
            if similar_question_id in list_of_bad_ids:
                list_of_questions.pop(-1)
                continue
            else:
                list_of_similar_questions.append(similar_question_id)

    return list_of_questions, list_of_similar_questions


def build_negative_sample_ids_only(all_question_ids, selected_questions_ids, similar_question_ids, post_parser_record):
    # this method returns a list of 550 question IDs that are not in the positive sample
    # provided by Professor Mansouri
    # 275 is the size of the positive data set, so 550 random questions will provide a
    # negative data set of the same size
    negative_sample = []
    while len(negative_sample) < 550:
        for q_id in range(len(all_question_ids)):
            if q_id not in selected_questions_ids and q_id not in similar_question_ids:
                try:
                    question_obj = post_parser_record.map_questions[q_id]
                except KeyError:
                    continue
                negative_sample.append(q_id)
    return negative_sample


def build_the_data_set_ids_only(q_list, similar_q_list, negative_q_list):
    # the q_list and similar_q_list should both be of length 275
    # the negative_q_list should be 550, double the size of the other two lists

    list_of_quadruples = []

    for x in range(len(q_list)):
        y = x - 1  # for indexing both q_list and similar_q_list and the first half of the negative_q_list
        w = y + 275  # for indexing the second half of the negative_q_list
        quadruple = (q_list[y], similar_q_list[y], negative_q_list[y], negative_q_list[w])
        list_of_quadruples.append(quadruple)

    return list_of_quadruples


def data_set_to_tsv_files_ids_only(list_of_quadruples):
    training_set = list_of_quadruples[:220]
    validation_set = list_of_quadruples[220:247]
    test_set = list_of_quadruples[247:]

    # writing the entire data set
    with open("feed_forward_data_set_ids.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in list_of_quadruples:
            writer.writerow(row)

    # writing the training data set
    with open("feed_forward_training_set_ids.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in training_set:
            writer.writerow(row)

    # writing the validation data set
    with open("feed_forward_validation_set_ids.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in validation_set:
            writer.writerow(row)

    # writing the test data set
    with open("feed_forward_test_set_ids.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in test_set:
            writer.writerow(row)


# function definitions section: get the embeddings
def get_title_plus_body(q_id, post_parser_record):
    question_obj = post_parser_record.map_questions[q_id]  # get the question object from the parser
    question_title = question_obj.title  # get the title
    clean_question_title = html_tag_remover(question_title)  # clean the title
    question_body = question_obj.body  # get the body
    clean_question_body = html_tag_remover(question_body)  # clean the body
    question_text = clean_question_title + clean_question_body  # concatenate
    return question_text  # return the concatenation


def parse_the_data_tsv_files(tsv_file, post_parser_record, model00):
    ff_nn_ready_data = []
    with open(tsv_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        for row in reader:  # each row has four numbers, the first two are a similar pair of
                            # questions, the third and forth are an unsimilar pair of questions

            pos_pair = []  # this will be a list of three things, [embedding01, embedding02, 1]
            neg_pair = []  # this will be a list of three things, [embedding01, embedding02, 0]

            # get the question IDs from the tsv file
            pos_q_pair_01 = int(row[0])
            pos_q_pair_02 = int(row[1])
            neg_q_pair_01 = int(row[2])
            neg_q_pair_02 = int(row[3])

            # get the question titles and bodies from the post_parser_record
            clean_text_pos_01 = get_title_plus_body(pos_q_pair_01, post_parser_record)
            clean_text_pos_02 = get_title_plus_body(pos_q_pair_02, post_parser_record)
            clean_text_neg_01 = get_title_plus_body(neg_q_pair_01, post_parser_record)
            clean_text_neg_02 = get_title_plus_body(neg_q_pair_02, post_parser_record)

            # get the question embeddings using the fasttext model
            pos_embedding_01 = get_sentence_embedding(model00, clean_text_pos_01)
            pos_embedding_02 = get_sentence_embedding(model00, clean_text_pos_02)
            neg_embedding_01 = get_sentence_embedding(model00, clean_text_neg_01)
            neg_embedding_02 = get_sentence_embedding(model00, clean_text_neg_02)

            # creating a list of the positive example
            # this list will be written into a TSV file later
            pos_pair.append(pos_embedding_01)
            pos_pair.append(pos_embedding_02)
            pos_pair.append(1)

            # creating a list of the negative example
            # this list will be written as a row in a TSV file later
            neg_pair.append(neg_embedding_01)
            neg_pair.append(neg_embedding_02)
            neg_pair.append(0)

            # appending the positive and negative example lists to a larger list of lists
            # this larger list of lists will be returned,
            # and used to write a TSV file
            ff_nn_ready_data.append(pos_pair)
            ff_nn_ready_data.append(neg_pair)

    return ff_nn_ready_data


def data_to_tsv_file_read_for_neural_network(list_of_lists, name_of_tsv_file_to_write):
    data = list_of_lists
    with open(name_of_tsv_file_to_write, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')

        # Write the header row
        writer.writerow(['q1_embedding', 'q2_embedding', 'similar_or_not'])

        # Write the data rows
        for row in data:
            writer.writerow(row)
# end of function definitions


# function calls
parsed_law_posts = PostParserRecord("Posts_law.xml")
# the_model = FastText.load('SeansFT.model')


# function calls section: make TSV files of the question IDs


# pos_examples_tsv_file = "duplicate_questions.tsv"
# the_question_ids, the_similar_question_ids = parse_the_tsv_file_ids_only(pos_examples_tsv_file)
#
# the_negative_question_ids = build_negative_sample_ids_only(
#     get_question_ids(parsed_law_posts),
#     the_question_ids,
#     the_similar_question_ids,
#     parsed_law_posts)
#
# the_quadruples = build_the_data_set_ids_only(
#     the_question_ids,
#     the_similar_question_ids,
#     the_negative_question_ids)
#
# data_set_to_tsv_files_ids_only(the_quadruples)


# function calls section: make TSV files of question embeddings and
#                     labels ready to use for neural network training,
#                     validation, and testing

the_model = FastText.load('SeansFT.model')

# training_data_lol = parse_the_data_tsv_files(
#     "feed_forward_training_set_ids.tsv",
#     parsed_law_posts,
#     the_model)
#
# validation_data_lol = parse_the_data_tsv_files(
#     "feed_forward_validation_set_ids.tsv",
#     parsed_law_posts,
#     the_model)
#
# test_data_lol = parse_the_data_tsv_files(
#     "feed_forward_test_set_ids.tsv",
#     parsed_law_posts,
#     the_model)
#
# data_to_tsv_file_read_for_neural_network(training_data_lol, "ready_to_use_training_data.tsv")
# data_to_tsv_file_read_for_neural_network(validation_data_lol, "ready_to_use_validation_data.tsv")
# data_to_tsv_file_read_for_neural_network(test_data_lol, "ready_to_use_test_data.tsv")

# end of function calls

