# SimpleText Task 1

This repository includes all of the programs and optimal results for my team's (Team Fletcher, one member, me) submission for SimpleText Task 1.

## Table of Contents

- [Installation](#Installation)
- [Text-Files](#Text-Files)
- [Steps-to-Run](#Steps-to-Run)
- [Model-and-Results](#Model-and-Results)
- [Baseline-Results](#Baseline-Results)
- [Our-Results](#Our-Results)
- [Conclusion](#Conclusion)

## Installation

To run this code, you will need access to the both the training data for task 1 of SimpleText CLEF lab and their servers ( via a user name and password). I will be assuming you have access for the rest of the explanation.

The necessary installs for the code are:

    torch
    tqdm
    sentence_transformers
    transformers
    ranx
    sklearn
    csv
    json
    os
    markdown
    requests
    bs4
    yake
    re
    sys

You can install them using pip:

    pip install torch tqdm sentence_transformers transformers ranx sklearn csv json os markdown requests bs4 yake re sys
    
Clone the repository all to one folder to properly run. You will need to add file paths to your own directories.

## Text Files
- SP12023topics.csv -> 6 columns (topic_id;topic_text;topic_url;query_id;query_text;abstract_url)


## Steps to Run
- Get the SimpleText dataset from CLEF


main.py
- All function calls are in main.py
- You will need to add three directory paths at the top of main.py
- These directories will be where you save your downloaded json files of scientific articles.
- You will need to add the file path to the SimpleText dataset.
IMPORTANT!
I do not suggest running the whole file at once. Look at each section and run the sections you need sepperately.

There are sections that focus on downloading relevant json files of scientific abstracts.

There are sections that make calls using python's requests library

There are sections that use YAKE to extract keywords from various sources.

There are sections that save dicionaries as text documents that can be passed to the evauation.py file using the terminal 

    python evaluation.py results_to_run
    
There are sections that prepare data, and then fine-tune a sentence bert model. (that was never tested)


## Model and Results

My final results come from comparing lists of extracted keywords from the news articles and the scientific abstracts against eachother. I compare the keyword lists using only YAKE extracted keywords (YAKE), YAKE extracted keywords with the query term added to the keyword list for the nes article (YAKE + QT), and then again with adding the keywords scraped from the news articles' websites (YAKE + QT + GS).


## Results

Baseline
NDCG@10 score: 0.399
MAP score: 0.461
~*~
YAKE
NDCG@10 score: 0.092
MAP score: 0.131
~*~
YAKE + QT 
NDCG@10 score: 0.107
MAP score: 0.122
~*~
YAKE + QT + GS
NDCG@10 score: 0.133
MAP score: 0.133


## Conclusion

As can be seen in the results above, while adding the keywords from the websites does better than using YAKE keywords all by themselves. The results are all significantly lower tham the baseline results of using the first 100 returned abstracts from using the query term.


