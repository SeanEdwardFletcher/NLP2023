# Sean Fletcher
# Assignment 01 - Question 01
# NLP spring 2023

# imports
import matplotlib.pyplot as plt
from collections import Counter
import re
# end of imports


# opening the .xml file
with open("Posts_Coffee.xml", "r") as file:
    xml_string = file.read()

# regex pattern to find the titles
title_grab_pattern = 'Title=".*Tags="'
regex_title_grab = re.compile(title_grab_pattern)

# using the regex pattern to get the titles
matches = regex_title_grab.findall(xml_string)


def title_cleaner(str_input):
    # this function strips the "Title=" and "Tags=" from each regex match
    return str_input[7:-8]


list_of_titles = [title_cleaner(i) for i in matches]


# count the frequency of each word in the list of strings
word_counts = Counter(" ".join(list_of_titles).split())

# sort the dictionary by word frequency
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# extract the rank and frequency of each word
ranks = range(1, len(sorted_word_counts) + 1)
frequencies = [count for word, count in sorted_word_counts]

# plot the data
plt.scatter(ranks, frequencies)
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Zipf's Law Graph of Coffee Post Titles")
plt.show()
