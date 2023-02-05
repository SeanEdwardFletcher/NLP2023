# Sean Fletcher
# Assignment 01 - Question 02
# NLP spring 2023

# imports
import nltk
from nltk import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
# nltk.download("stopwords")
# end of imports


with open("Posts_Coffee.xml", "r") as file:
    xml_string = file.read()

# a regex pattern to grab the Titles by identifying the tags before and after each title
title_grab_pattern = 'Title=".*Tags="'
regex_title_grab = re.compile(title_grab_pattern)  # the regex object
matches = regex_title_grab.findall(xml_string)  # the strings in the XLM doc that match the pattern


def title_cleaner(str_input):
    # this function strips the 'Title="' and 'Tags="' from each regex match
    return str_input[7:-8]


# cleaning the tags off of the regex matches
list_of_titles = [title_cleaner(i) for i in matches]

# All the titles in one string for processing
big_string_of_titles = ' '.join(list_of_titles)


def punct_removal(input_string):
    # removes the punctuation examples in list_of_punct from the input_string
    # returns the punctuation-free string
    # I build it myself because I didn't like the data I was getting with the
    # built-in punctuation cleaner.
    list_of_punct = ["!", "@", "#", "$", "%", "^",
                     "&", "*", "(", ")", "[", "]",
                     "'", '"', "'", ";", ":", "{",
                     "}", "\\", ",", ".", "?", "/",
                     "-", "+", "="]
    clean_string = input_string
    for char in list_of_punct:
        clean_string = clean_string.replace(char, " ")
    return clean_string


# removing the punctuation from the titles
clean_big_string_of_titles = punct_removal(big_string_of_titles)

# breaking the huge string of titles into individual words
raw_word_tokens = word_tokenize(clean_big_string_of_titles)

# making sure each word is lower case for word counting purposes
clean_word_tokens = [word.lower() for word in raw_word_tokens]


# removing the stopwords from the clean_word_tokens list of strings
stopwords_list = stopwords.words("english")
clean_word_tokens_no_stopwords = []
for word in clean_word_tokens:
    if word not in stopwords_list:
        clean_word_tokens_no_stopwords.append(word)


# this is a set of strings to exclude from the word cloud (I'm excluding no strings)
# this is necessary because the word cloud generator automatically removes stop words
# unless you provide your own list of stopwords to remove. This is my life. It's pretty short
stop_words = {""}


# Convert word lists to single strings for making word clouds
word_cloud_string = " ".join(clean_word_tokens)
word_cloud_string_no_stopwords = " ".join(clean_word_tokens_no_stopwords)

# generating the word clouds
word_cloud = WordCloud(background_color="white", stopwords=stop_words, max_words=20).generate(word_cloud_string)
word_cloud_no_stopwords = WordCloud(background_color="white", stopwords=stop_words, max_words=20).generate(word_cloud_string_no_stopwords)

# viewing the wordclouds
# plt.imshow(word_cloud, interpolation='bilinear')
plt.imshow(word_cloud_no_stopwords, interpolation='bilinear')
plt.axis("off")
plt.show()

