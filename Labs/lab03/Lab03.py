
# NLP 2023
# Lab 03
# Faranak JahediBashiz, Sean Fletcher


# imports
from bs4 import BeautifulSoup
from post_parser_record_lab03 import PostParserRecord
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
from nltk import collections
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download("punkt")
# end of imports


# global variables
post_file_path = "Law_Posts.xml"
law_posts_record = PostParserRecord(post_file_path)
post_reader = PostParserRecord(post_file_path)
stop_words = set(stopwords.words('english'))
limit = 5
# end of global variables


# functions
def find_frequent_tag(post_reader_object, tag_limit):
    """
    this function takes in a post_reader and returns the most frequent tags
    :param post_reader_object:
    :param tag_limit: the amount of most frequent tags you want
    :return: list: a list of strings of the most frequent tags
    """
    lst_all_tags = []
    # iterates through the post
    for question_id in post_reader_object.map_questions:
        question = post_reader_object.map_questions[question_id]
        # current question tags
        tags = question.tags
        for tag in tags:
            lst_all_tags.append(tag)
    # makes a count of the tags
    tag_counter = Counter(lst_all_tags)
    # finds the most frequent tags
    most_common_tags = tag_counter.most_common(tag_limit)
    # makes a list of the most frequent tags
    lst_common_tags = [tag[0] for tag in most_common_tags]
    return lst_common_tags


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
                     "-", "+", "=", "<", ">"]
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
    no_html_string = html_tag_remover(input_string)
    clean_string = punct_removal(no_html_string)
    return clean_string


def string_stemmer(input_string):
    """
    a stemmer removes the "stems" from word strings.
    examples:
         runner --> run
         running --> run
         run --> run
         loving --> lov      it's not very good with some words
         lover --> lov
         was --> wa
    :param input_string:
    :return: string: the input_string after PorterStemmer() does its thing to it
    """

    tokenized_text = nltk.word_tokenize(input_string)

    stemmer = PorterStemmer()

    stemmed_text = [stemmer.stem(word) for word in tokenized_text]

    return " ".join(stemmed_text)


def build_a_wordcloud(string_of_words, size, remove_stopwords=True, use_stemmer=False):
    """

    :param use_stemmer: uses the NLTK stemmer to stem the words from the input
    :param string_of_words: the string you want a frequency-based wordcloud of
    :param size: how many of the most frequent words do you want included?
    :param remove_stopwords: if True, stopwords will be removed
    :return: N/A...  it makes a wordcloud... ...
    """
    # turn the input string into a list of words
    word_list = string_of_words.split()

    # removes the stopwords if variable remove_stopwords is True
    if remove_stopwords:
        stp_wrds = set(stopwords.words('english'))
        word_list = [word for word in word_list if word.lower() not in stp_wrds]

    if use_stemmer:
        word_list = [string_stemmer(word) for word in word_list]

    # Create a frequency distribution of the words
    word_frequency = collections.Counter(word_list)

    # Get the most common words
    most_common_words = word_frequency.most_common(size)

    # Get JUST the words
    most_common_words_list = [word[0] for word in most_common_words]

    # Join the filtered words into a string
    the_wordcloud_string = ' '.join(most_common_words_list)

    # this is a set of strings to exclude from the word cloud (I'm excluding no strings)
    # this is necessary because the word cloud generator automatically removes stop words
    # unless you provide your own list of stopwords to remove. Since I already have a parameter
    # dealing with this above, when I call the WordCloud generator I'm going to NOT remove any
    # stopwords in that call. You can see stopwords=tiny_list in the call below
    tiny_list = {""}

    word_cloud = WordCloud(background_color="white", stopwords=tiny_list).generate(the_wordcloud_string)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def plot_frequency_distribution(input_spring, question_tag_name):
    """

    :param input_spring: the string whose word frequency you want plotted
    :param question_tag_name: the name of the tag the questions have in common
    :return:
    """

    question_tag_name = question_tag_name  # this line might be irrelevant

    # count the frequency of each word in the string
    word_counts = Counter(input_spring.split())

    # sort the dictionary by word frequency
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # extract the rank and frequency of each word
    ranks = range(1, len(sorted_word_counts) + 1)
    frequencies = [count for word, count in sorted_word_counts]

    # plot the data
    plt.scatter(ranks, frequencies)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title(f"Zipf's Law Graph of question tag: {question_tag_name}")
    plt.show()


def question_bodies_by_tag(list_of_tags):
    """
    this function iterates through a post_reader object
    if a question in that object has a tag that is in the list_of_tags argument
    it appends that question's body to a string
    :param list_of_tags:
    :return: String:
    """
    string_of_question_bodies = ""
    for question_id in post_reader.map_questions:
        question = post_reader.map_questions[question_id]
        # current question body
        text = question.body
        # current question tags
        lst_tags = question.tags
        for tag in lst_tags:
            if tag in list_of_tags:
                string_of_question_bodies += text
    return string_of_question_bodies
# end of functions


# function calls
five_most_common_tags = find_frequent_tag(law_posts_record, 5)
tag01 = [five_most_common_tags[0]]
tag02 = [five_most_common_tags[1]]
tag03 = [five_most_common_tags[2]]
tag04 = [five_most_common_tags[3]]
tag05 = [five_most_common_tags[4]]


# # Lab03 Step 02 call
# big_string_of_question_bodies = question_bodies_by_tag(five_most_common_tags)
# cleaned_string = string_cleaner(big_string_of_question_bodies)
# build_a_wordcloud(cleaned_string, 30)

# # Lab03 Step 03 call
# big_string_of_question_bodies = question_bodies_by_tag(five_most_common_tags)
# cleaned_string = string_cleaner(big_string_of_question_bodies)
# build_a_wordcloud(string_stemmer(cleaned_string), 30, use_stemmer=True)


# Lab03 Step 04 calls

# tag 01 frequency distribution call
# tag01_Q_body = question_bodies_by_tag(tag01)
# clean_tag01_string = string_cleaner(tag01_Q_body)
# plot_frequency_distribution(clean_tag01_string, tag01[0])

# tag 02 frequency distribution call
# tag02_Q_body = question_bodies_by_tag(tag02)
# clean_tag02_string = string_cleaner(tag02_Q_body)
# plot_frequency_distribution(clean_tag02_string, tag02[0])

# tag 03 frequency distribution call
# tag03_Q_body = question_bodies_by_tag(tag03)
# clean_tag03_string = string_cleaner(tag03_Q_body)
# plot_frequency_distribution(clean_tag03_string, tag03[0])

# tag 04 frequency distribution call
# tag04_Q_body = question_bodies_by_tag(tag04)
# clean_tag04_string = string_cleaner(tag04_Q_body)
# plot_frequency_distribution(clean_tag04_string, tag04[0])

# tag 05 frequency distribution call
# tag05_Q_body = question_bodies_by_tag(tag05)
# clean_tag05_string = string_cleaner(tag05_Q_body)
# plot_frequency_distribution(clean_tag05_string, tag05[0])

# end of function calls
