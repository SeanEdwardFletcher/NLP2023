
# NLP Assignment 02 Sean Fletcher

# imports
from ass_02_functions import *
from ass_02_NBModel import NBModel
from post_parser_record_ass02 import PostParserRecord
# end of imports


# parsing the XML file
parsed_law_posts = PostParserRecord("Posts_law_ass02.xml")

# getting the 21 most frequent tags
frequent_tag_lst = get_frequent_tags(parsed_law_posts)

# removing "contract" as a tag because it has no posts after 2021
frequent_tag_lst.remove("contract")

# separating the posts from the parsed XML file into training and testing sets
# the test and training dicts have a list of tuples for each key,
# the first tuple element is the title, the second is the body
test_dict, training_dict = build_train_test(parsed_law_posts, frequent_tag_lst)

# finding the "class prior probabilities" for each of the 20 question tags
dict_class_prior_probabilities = find_class_prior_probabilities(training_dict)

# cleaning the data from the training and test sets
clean_titles_training_set, clean_bodies_training_set, clean_training_set = clean_the_data_sets(training_dict)
clean_titles_test_set, clean_bodies_test_set, clean_test_set = clean_the_data_sets(test_dict)
# end of cleaning

# below are the function calls that train a Naive Bayes Model, test that model, and calculate the F1 scores
# uncomment one section at a time because it takes a few seconds to run.

# LawPostsBodies = NBModel(clean_bodies_training_set, dict_class_prior_probabilities)
# bodies_f1_scores = LawPostsBodies.get_the_f1_scores(clean_bodies_test_set)
# print("Question Bodies F1 Scores:")
# for key, value in bodies_f1_scores.items():
#     print(key, value)

# LawPostsTitles = NBModel(clean_titles_training_set, dict_class_prior_probabilities)
# titles_f1_scores = LawPostsTitles.get_the_f1_scores(clean_titles_test_set)
# print("Question Titles F1 Scores:")
# for key, value in titles_f1_scores.items():
#     print(key, value)

# LawPosts = NBModel(clean_training_set, dict_class_prior_probabilities)
# f1_scores = LawPosts.get_the_f1_scores(clean_test_set)
# print("Question Titles and Bodies F1 Scores:")
# for key, value in f1_scores.items():
#     print(key, value)
