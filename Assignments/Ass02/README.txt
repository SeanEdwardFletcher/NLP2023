This repository contains:
function_calls
functions
NBModel
chat_gpt_api_call
Post
post_parser_record
README (this file)

This collection of code is for the second assignment in a Natural Language
Processing class.

When running the code, use function_calls. At the bottom of function_calls 
are three blocks of code that have been commented out. Each block corosponds 
to training a Naive Bayes Classifier (NBModel) with one of three training data sets,
and then tests that NBModel with testing data sets.

functions: a file with functions that do a variety of tasks, but mostly functions
that are used to prepare the training and data sets that will be using in the
NBModel

NBModel: there is a collection of functions and a class. The functions are used by
the class, which is why there are in the same file.

chat_gpt_api_call: This is python code that asks a question to Chat GPT, it's
one of the requirments for the second assignment.

Post & post_parser_record: These files are used to process XML files downloaded from
Stack Exchange. This is where the data was collected that was used to train and test
the NBModel
