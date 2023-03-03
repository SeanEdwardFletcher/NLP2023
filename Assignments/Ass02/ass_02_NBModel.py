# NLP Assignment 02 Sean Fletcher

from collections import Counter
import numpy as np


def count_frequency(input_string):
    # count the frequency of each word in the string
    word_counts = Counter(input_string.split())
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_counts


def count_all_words(input_frequency_list):
    # tallies all tokens in the model
    word_count = 0
    for tupl in input_frequency_list:
        word_count += tupl[1]
    return word_count


def add_one_to_count_frequencies(input_frequency_list):
    # applies +1 smoothing to the model
    smooth_frequency_list = []
    for tupl in input_frequency_list:
        smooth_tupl = (tupl[0], tupl[1] + 1)
        smooth_frequency_list.append(smooth_tupl)
    return smooth_frequency_list


def make_token_probability_list(total_num_of_tokens, input_frequency_list):
    # returns a dictionary with the token and that token's probability
    the_model_dict = {}
    for tupl in input_frequency_list:
        the_model_dict[tupl[0]] = tupl[1] / total_num_of_tokens

    return the_model_dict


def count_frequency_by_class(input_dictionary):
    """
    :param input_dictionary: this python dictionary needs to have each key mapped
                            to a single value [a list]. That list needs to be a list of
                            strings.

    :return: a dictionary with the classes/tags as keys and a list of two-element tuples
                 a string of the word
                 an integer of how many times that word occurred in the class/tag
    """
    count_frequency_dict = {}
    for key in input_dictionary.keys():
        bag_of_words_in_class = " ".join(input_dictionary[key])
        class_count_frequency = count_frequency(bag_of_words_in_class)
        count_frequency_dict[key] = class_count_frequency
    return count_frequency_dict


def log_probability_summation(list_of_probabilities, class_prior):
    # multiplies the probabilities in log space
    first_prob = list_of_probabilities[0]
    running_log = np.log(first_prob)
    for prob in list_of_probabilities[1:]:
        log_prob = np.log(prob)
        running_log += log_prob
    running_log += np.log(class_prior)

    return np.exp(running_log)


def find_the_highest_probability(dict_of_probs):
    highest_key = max(dict_of_probs.items(), key=lambda x: x[1])[0]
    return highest_key


def fill_the_matrix(input_dict):
    # creates a confusion matrix based on the model's predictions
    list_of_dicts = []
    for key in input_dict.keys():
        tag_dict = {"tag": key, "TP": 0, "FN": 0, "FP": 0, "TN": 0}
        list_of_dicts.append(tag_dict)
    for dict00 in list_of_dicts:
        real_label = dict00["tag"]
        for key in input_dict.keys():
            if key != real_label:
                for label in input_dict[key]:
                    if key == label:
                        dict00["FP"] += 1
                    else:
                        dict00["TN"] += 1
            else:
                for label in input_dict[key]:
                    if key == label:
                        dict00["TP"] += 1
                    else:
                        dict00["FN"] += 1
    return list_of_dicts


class NBModel:

    def __init__(self, input_dictionary, class_prior_probabilities):

        self.class_prior_probabilities = class_prior_probabilities

        self.bag_of_words = ""
        for key in input_dictionary.keys():
            self.bag_of_words += " ".join(input_dictionary[key])

        self.number_of_token_types = len(count_frequency(self.bag_of_words))
        self.number_of_tokens = count_all_words(count_frequency(self.bag_of_words))

        self.frequency_distribution_by_class = count_frequency_by_class(input_dictionary)

        self.smooth_frequency_distribution_by_class = {}
        for key in input_dictionary.keys():
            self.smooth_frequency_distribution_by_class[key] = add_one_to_count_frequencies(
                self.frequency_distribution_by_class[key])

        self.smooth_number_of_tokens = self.number_of_tokens + self.number_of_token_types

        self.token_probabilities_by_class = {}
        for key in input_dictionary.keys():
            self.token_probabilities_by_class[key] = make_token_probability_list(
                self.smooth_number_of_tokens,
                self.smooth_frequency_distribution_by_class[key]
            )

    def document_to_dict_of_probs(self, input_string):
        # this method creates a dictionary for each document
        # where the keys are the tags and the values
        # are the probabilities of this doc having that tag

        # Get a list of keys from the class' token probabilities dictionary
        keys_list = list(self.token_probabilities_by_class.keys())
        # Create a new dictionary with the same keys
        new_dict = {key: [] for key in keys_list}

        list_of_tokens = input_string.split()

        for token in list_of_tokens:
            for key in self.token_probabilities_by_class:
                if token in self.token_probabilities_by_class[key]:
                    new_dict[key].append(self.token_probabilities_by_class[key][token])

        for key in new_dict.keys():
            list_of_probs = new_dict[key]
            if len(list_of_probs) == 0:  # this if:else is here because something weird was happening with titles
                pass
            else:
                log_summation = log_probability_summation(list_of_probs, self.class_prior_probabilities[key])
                new_dict[key] = log_summation

        return new_dict

    def test(self, input_dictionary):
        # this method takes the test data and tests it against the model
        # it returns a dictionary where the keys are the tags,
        # and the values are lists of dictionaries
        # where each dictionary in the list represents a document that was tested and has
        # that document's probabilities for each tag

        # Get a list of keys from the input dictionary and make a new dictionary
        keys_list = list(input_dictionary.keys())
        dict_to_return = {key: [] for key in keys_list}

        for key in input_dictionary.keys():
            for doc in input_dictionary[key]:  # each doc is a string
                dict_of_doc_probs = self.document_to_dict_of_probs(doc)
                dict_to_return[key].append(dict_of_doc_probs)

        return dict_to_return

    def calculate_accuracy(self, input_dictionary):
        # this method returns a confusion matrix for the test data
        # provided to it

        probability_by_classes = self.test(input_dictionary)

        keys_list = list(probability_by_classes.keys())
        dict_of_results = {key: [] for key in keys_list}

        for key in probability_by_classes.keys():
            list_of_dicts = probability_by_classes[key]
            for dict00 in list_of_dicts:
                key_with_max_value = max(dict00, key=dict00.get)
                dict_of_results[key].append(key_with_max_value)

        confusion_matrix = fill_the_matrix(dict_of_results)
        return confusion_matrix

    def get_the_f1_scores(self, input_dictionary):
        # this method uses the following formulas to calculate the f1 scores
        # for each class(tag) in the model, it also has a "macro" key that
        # has the macro f1 score

        # F1 = 2PR/P+R
        # P = TP/TP+FP
        # R = TP/TP+FN

        confusion_matrix = self.calculate_accuracy(input_dictionary)

        keys_list = list(input_dictionary.keys())
        dict_of_f1_scores = {key: 0 for key in keys_list}

        for dict00 in confusion_matrix:
            key = dict00["tag"]
            p = dict00["TP"]/(dict00["TP"] + dict00["FP"])
            r = dict00["TP"]/(dict00["TP"] + dict00["FN"])
            f1 = 0
            if p == 0.0 and r == 0.0:
                dict_of_f1_scores[key] = f1
            else:
                f1 = (2*p*r)/(p+r)
                dict_of_f1_scores[key] = f1

        running_total = 0

        for key in dict_of_f1_scores.keys():
            running_total += dict_of_f1_scores[key]

        f1_macro = running_total/20

        dict_of_f1_scores["macro"] = f1_macro
        return dict_of_f1_scores



