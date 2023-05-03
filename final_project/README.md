
To run this project:

main.py
All function calls are in main.py
You will need to add three directory paths at the top of main.py
These directories will be where you save your downloaded json files of scientific articles.
You will need to add the file path to your training data.

download_elastic.py
You will need to add a directory path and your user name and password to the file download_elastic.py

evaluation.py
Pass the text files created from some of the dictionaries to this file using your terminal to get your scores
ex. ->  python evaluation.py text_file_to_evaluate


In main.py I do not suggest running the whole file at once. Look at each section and run the sections you need sepperately.


There are sections that focus on downloading relevant json files of scientific abstracts.

There are sections that make calls using python's requests library

There are sections that use YAKE to extract keywords from various sources.

There are sections that save dicionaries as text documents that can be passed to the evauation.py file using the terminal 

There are sections that prepare data, and then fine-tune a sentence bert model.

