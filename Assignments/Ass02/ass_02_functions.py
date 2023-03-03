# imports
from post_parser_record_ass02 import PostParserRecord
from collections import Counter
from bs4 import BeautifulSoup
# end of imports


# global variables
#
# end of global variables


# functions
def punct_removal(input_string):
    """
    I build it myself because I didn't like the data I was getting with the
    built-in punctuation cleaners.
    :param input_string:
    :return: string: a string without punctuation
    """
    list_of_punct = ["!", "@", "#", "$", "%", "^",
                     "&", "*", "(", ")", "[", "]",
                     "'", '"', "'", ";", ":", "{",
                     "}", "\\", ",", ".", "?", "/",
                     "-", "+", "=", "<", ">", "—"]
    clean_string = input_string
    for char in list_of_punct:
        clean_string = clean_string.replace(char, " ")
    return clean_string


def html_tag_remover(input_string):
    """
    this function uses BeautifulSoup to remove the HTML element tags from a string
    :param input_string:
    :return: text: a string without the HTML element tags
    """
    soup = BeautifulSoup(input_string, "html.parser")
    text = soup.get_text()
    return text


def string_cleaner(input_string):
    """
    this function runs input_string through:
         html_tag_remover()
         punct_removal()
    :param input_string:
    :return: string: the input string after running it through the two functions
    """
    no_html_string = html_tag_remover(input_string)  # do this before removing punctuation, obviously
    clean_string = punct_removal(no_html_string)
    return clean_string.lower()


# Getting the top-20 frequent tags in LawSE -- There is a reason for passing 21
def get_frequent_tags(post_parser, topk=21):
    lst_tags = []
    for question_id in post_parser.map_questions:
        question = post_parser.map_questions[question_id]
        creation_date_year = int(question.creation_date.split("-")[0])
        tag = question.tags[0]
        lst_tags.append(tag)
    tag_freq_dic = dict(Counter(lst_tags))
    tag_freq_dic = dict(sorted(tag_freq_dic.items(), key=lambda item: item[1], reverse=True))
    return list(tag_freq_dic.keys())[:topk]


# Getting dictionary of train and test samples in form of
# key: tag value: list of tuples in form of (title, body)
def build_train_test(post_parser, lst_frequent_tags):
    dic_training = {}
    dic_test = {}
    for question_id in post_parser.map_questions:
        question = post_parser.map_questions[question_id]
        creation_date_year = int(question.creation_date.split("-")[0])
        tag = question.tags[0]
        if tag in lst_frequent_tags:
            title = question.title
            body = question.body
            if creation_date_year > 2021:
                if tag in dic_test:
                    dic_test[tag].append((title, body))
                else:
                    dic_test[tag] = [(title, body)]
            else:
                if tag in dic_training:
                    dic_training[tag].append((title, body))
                else:
                    dic_training[tag] = [(title, body)]
    return dic_test, dic_training


def clean_the_data_sets(input_dictionary):
    """
    this function is built to work with the dictionaries returned by build_train_test()
    it:
        extracts the strings embedded in the values,
        cleans it by calling string_cleaner()
        stores the cleaned strings in two new dictionaries
        returns those two dictionaries

    :param input_dictionary: this python dictionary needs to have each key mapped
                            to a single value [a list]. That list needs to be a list of
                            tuples. In each tuple there are two elements, both are strings.
                            The first string is the title of the post, the second string
                            is the body of the post

    :return: two dictionaries. The Keys are the keys from input_dictionary and each key have
                            one value, a list. That list is a list of strings. In one dictionary
                            the strings are the cleaned titles of the posts, in the other dictionary
                            the strings are the cleaned bodies of the posts.

    """
    clean_titles_by_tag = {}
    clean_bodies_by_tag = {}
    clean_posts_by_tag = {}
    for key in input_dictionary.keys():
        for tupe in input_dictionary[key]:

            cleaned_title = string_cleaner(tupe[0])
            cleaned_body = string_cleaner(tupe[1])
            cleaned_post = cleaned_title + cleaned_body

            if key in clean_titles_by_tag:
                clean_titles_by_tag[key].append(cleaned_title)
            else:
                clean_titles_by_tag[key] = [cleaned_title]

            if key in clean_bodies_by_tag:
                clean_bodies_by_tag[key].append(cleaned_body)
            else:
                clean_bodies_by_tag[key] = [cleaned_body]

            if key in clean_posts_by_tag:
                clean_posts_by_tag[key].append(cleaned_post)
            else:
                clean_posts_by_tag[key] = [cleaned_post]

    return clean_titles_by_tag, clean_bodies_by_tag, clean_posts_by_tag


def make_a_bag_of_words(list_of_strings):
    return " ".join(list_of_strings)


def count_frequency(input_string):
    # count the frequency of each word in the string
    word_counts = Counter(input_string.split())

    # sort the dictionary by word frequency
    # this is a list of 2-element tuples with the string form of the word as the
    # first element and an integer of how many times that word appeeared as the
    # second element
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    return sorted_word_counts


def count_frequency_by_class(input_dictionary):
    """

    :param input_dictionary: this python dictionary needs to have each key mapped
                            to a single value [a list]. That list needs to be a list of
                            strings.


    :return: a dictionary with the classes/tags as keys and a list of tuples with two elements each
                 a string of the word
                 an integer of how many times that word occured in the class/tag
    """
    count_frequency_dict = {}
    for key in input_dictionary.keys():
        bag_of_words_in_class = make_a_bag_of_words(input_dictionary[key])
        class_count_frequency = count_frequency(bag_of_words_in_class)
        count_frequency_dict[key] = class_count_frequency
    return count_frequency_dict


def find_class_prior_probabilities(input_dictionary):
    total_number_of_posts = 0
    number_of_posts_by_tag = {}
    prior_class_probabilities = {}

    for key in input_dictionary.keys():
        posts_per_tag = len(input_dictionary[key])
        number_of_posts_by_tag[key] = posts_per_tag
        total_number_of_posts += posts_per_tag

    for key in number_of_posts_by_tag.keys():
        class_prior_probability = number_of_posts_by_tag[key] / total_number_of_posts
        prior_class_probabilities[key] = class_prior_probability

    return prior_class_probabilities



