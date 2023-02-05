# Sean Fletcher, Aimee Haus, SJ Franklin
# COS 470 - Natural Language Processing
# Lab 2 - February 3, 2023


# imports
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os
import collections
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# end of imports


# the functions


def traverse_directory(directory_path):
    """
     # this function recursively searches a directory and matches a regex against the file names
     # if the file name begins with a digit, it appends that file's path to a list
     # it returns that list

    :param directory_path:
    :return: List: a list of file paths
    """
    list_of_file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if re.match(r"^\d", file):
                full_path = os.path.join(root, file)
                list_of_file_paths.append(full_path)

    return list_of_file_paths


def grab_the_dates(input_string):
    """
    this function grabs every substring that matches with the regex
    :param input_string:
    :return: list: a list of strings the regex matched with
    """

    # regex strings for "month date", "date month", "month year", and "year month"
    mon_date_regex_str = '(January|February|March|April|May|June|July|August|September|October|November|December) [0-3]?[0-9]'
    date_mon_regex_str = '[0-3]?[1-9] (January|February|March|April|May|June|July|August|September|October|November|December)'
    mon_year_regex_str = '(January|February|March|April|May|June|July|August|September|October|November|December) (\d{4,4})'
    year_mon_regex_str = '(\d{4,4})+ (January|February|March|April|May|June|July|August|September|October|November|December)'

    # concatenating the four regex strings into one big regex
    # using (regex01|regex02|regex03|regex04) syntax
    big_regex = "(" + mon_date_regex_str + \
                "|" + date_mon_regex_str + \
                "|" + mon_year_regex_str + \
                "|" + year_mon_regex_str + \
                ")"

    big_regex_object = re.compile(big_regex)
    return big_regex_object.findall(input_string)




def date_reformatter(input_string):
    """

    :param input_string:
    :return:
    """
    # regex strings for "month date", "date month", "month year", and "year month"
    mon_date_regex_str = '(January|February|March|April|May|June|July|August|September|October|November|December) [0-3]?[0-9]'
    date_mon_regex_str = '[0-3]?[1-9] (January|February|March|April|May|June|July|August|September|October|November|December)'
    mon_year_regex_str = '(January|February|March|April|May|June|July|August|September|October|November|December) (\d{4,4})'
    year_mon_regex_str = '(\d{4,4})+ (January|February|March|April|May|June|July|August|September|October|November|December)'

    # these variables are boolean
    mon_date_match = re.search(mon_date_regex_str, input_string)
    date_mon_match = re.search(date_mon_regex_str, input_string)
    mon_year_match = re.search(mon_year_regex_str, input_string)
    year_mon_match = re.search(year_mon_regex_str, input_string)

    # elifs to rout input formats to the desired format mm/dd/yyyy
    if not mon_date_match and not date_mon_match and not mon_year_match and not year_mon_match:
        print("The input string did not match our target structures.")
        print(input_string)
        # Maybe we want to collect items that end up here? This is just a lab though, so maybe don't go too crazy
        return "00/00/0000"
    elif mon_date_match:
        if input_string == "February 29":
            return "02/29/2020"
        datetime_obj = datetime.strptime(input_string, "%B %d")
        formatted_date = datetime_obj.strftime("%m/%d/2020")
        return formatted_date
    elif date_mon_match:
        if input_string == "29 February":
            return "02/29/2020"
        if input_string == "31 April":
            return "04/30/2020"
        if input_string == "33 July":
            return "07/31/2020"
        datetime_obj = datetime.strptime(input_string, "%d %B")
        formatted_date = datetime_obj.strftime("%m/%d/2020")
        return formatted_date
    elif mon_year_match:
        datetime_obj = datetime.strptime(input_string, "%B %Y")
        formatted_date = datetime_obj.strftime("%m/00/2020")
        return formatted_date
    elif year_mon_match:
        datetime_obj = datetime.strptime(input_string, "%Y %B")
        formatted_date = datetime_obj.strftime("%m/00/%Y")
        return formatted_date
    else:
        print("Something fishy happened... this shouldn't print")
        return "00/00/0000"


def file_content_to_string(file_name):
    """
    :param file_name: string
    :return: string
    """
    with open(file_name, "r", encoding = "utfa") as file:
        return file.read()


def grab_numbers_and_surrounding_words(input_string):
    """
    :param input_string:
    :return: list: a list of strings the regex matched with
    """
    # this regex grabs a word-number-word pattern
    regex_pattern_01 = "[A-Za-z]+ ?[0-9]+ ?[A-Za-z]+"
    regex_object_01 = re.compile(regex_pattern_01)
    return regex_object_01.findall(input_string)


def remove_the_numbers(input_string):
    """
    :param input_string: String
    :return: String: the input string without numbers

    the regex '\d+' finds the numbers in the string and .sub() replaces them with ' '
    """
    return re.sub(r'\d+', ' ', input_string)


def build_a_wordcloud(string_of_words):
    """
    This function builds a 20-word wordcloud out of the string passed to it

    ... it might break if the string passed into has fewer than 100 unique words...
    ... I haven't tested that...
    
    :param string_of_words:
    :return: N/A
    """
    # Get a list of English stopwords
    stop_words = set(stopwords.words('english'))

    # turn the input string into a list of words
    word_list = string_of_words.split()

    # Create a frequency distribution of the words
    word_frequency = Counter(word_list)

    # Get the 100 most common words
    most_common_words = word_frequency.most_common(100)

    # Get JUST the words
    most_common_words_list = [word[0] for word in most_common_words]

    # Remove stopwords from the list
    filtered_words = [word for word in most_common_words_list if word.lower() not in stop_words]

    # top 20 words
    twenty_most_common_words = filtered_words[0:19]

    # Join the filtered words into a string
    the_wordcloud_string = ' '.join(twenty_most_common_words)

    word_cloud = WordCloud(background_color="white", max_words=20).generate(the_wordcloud_string)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def get_all_the_regex_matches(list_of_file_paths, function):
    """
    this function:
       takes a list of file paths
       calls file_content_to_string() which opens & reads each file, returning a string
       calls a regex filtering function that searches each string with a regex
       returns the regex matches as a list of strings
    :param list_of_file_paths:
    :param function: use either   grab_the_dates()   or   grab_numbers_and_surrounding_words()
    :return: List: a list of strings
    """
    big_list_of_matches = []
    for file_path in list_of_file_paths:
        file_data_as_string = file_content_to_string(file_path)
        list_of_match_strings = function(file_data_as_string)
        for match in list_of_match_strings:
            big_list_of_matches.append(match)
    return big_list_of_matches


# Runs all of our functions to display our dates
def run_step_02(directory_path):
    paths_list = traverse_directory(directory_path)

    # second param is: grab_the_dates
    # also, this returns tuples... which I wasn't expecting
    all_the_regex_matches = get_all_the_regex_matches(paths_list, grab_the_dates)

    # this gets the date we were looking for from the tuples
    list_of_raw_dates = []
    for match in all_the_regex_matches:
        list_of_raw_dates.append(match[0])

    # this reformats the dates
    list_of_formatted_dates = []
    for date in list_of_raw_dates:
        list_of_formatted_dates.append(date_reformatter(date))

    return list_of_formatted_dates


# Runs all of our functions to display our word cloud
def run_the_extra_credit(directory_path):
    paths_list = traverse_directory(directory_path)
    # second param is: grab_numbers_and_surrounding_words
    all_the_regex_matches = get_all_the_regex_matches(paths_list, grab_numbers_and_surrounding_words)
    all_the_regex_matches_no_numbers = [remove_the_numbers(match) for match in all_the_regex_matches]
    one_big_string = " ".join(all_the_regex_matches_no_numbers)
    build_a_wordcloud(one_big_string)


# end of functions


# let's run the things!

# change this string to the path of the directory you want to search
the_directory_path = "TLS-Covid19/txt"

# # call the Step 2 function
print(run_step_02(the_directory_path))

# call the extra credit function
run_the_extra_credit(the_directory_path)
